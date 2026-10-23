from torch_geometric.nn import TransformerConv
import torch
import torch.nn.functional as F

class LocalGraphTransformer(torch.nn.Module):
    def __init__(
            self,
            layers: int,
            attention_heads: int,
            input_channels: int,
            hidden_channels: int,
            classes: int
    ):  
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        
        self.layer_list = []
        self.norm_list = [] 
        
        self.head_dim = hidden_channels // attention_heads

        self.layer_list.append(
            TransformerConv(
                in_channels=input_channels,
                out_channels=self.head_dim,
                heads=attention_heads,
                concat=True
            )
        )
        self.norm_list.append(torch.nn.LayerNorm(hidden_channels))

        for layer_number in range(1, layers):
            if layer_number == layers - 1:
                self.layer_list.append(
                    TransformerConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=attention_heads,
                        concat=False
                    )
                )
                self.norm_list.append(torch.nn.LayerNorm(hidden_channels))
                break

            self.layer_list.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=self.head_dim,
                    heads=attention_heads,
                    concat=True
                )
            )
            self.norm_list.append(torch.nn.LayerNorm(hidden_channels))

        self.output_layer = torch.nn.Linear(
                in_features=hidden_channels,
                out_features=classes
        )

        self.layers = torch.nn.ModuleList(self.layer_list)
        self.norms = torch.nn.ModuleList(self.norm_list)

    def forward(self, x, edge_index):
        for layer, norm in zip(self.layers, self.norms):
            x = F.dropout(x, p=0.5, training=self.training)
            x = layer(x, edge_index)
            
            x = norm(x)
            x = self.leaky_relu(x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.output_layer(x)
        return x

class HybridGraphTransformer(torch.nn.Module):
    def __init__(
            self,
            layers: int,
            attention_heads: int,
            input_channels: int,
            hidden_channels: int,
            classes: int
    ):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        
        self.initial_gn_state = torch.nn.Parameter(torch.zeros(size=(1, input_channels)))
        
        self.head_dim = hidden_channels // attention_heads
        
        self.layer_list = torch.nn.ModuleList()
        self.gn_mlp_list = torch.nn.ModuleList()
        
        self.norm_list = torch.nn.ModuleList()
        self.gn_norm_list = torch.nn.ModuleList()
        
        current_x_dim = input_channels
        current_gn_dim = input_channels

        for layer_idx in range(layers):
            is_last_layer = (layer_idx == layers - 1)
            
            if is_last_layer:
                out_dim_conv = hidden_channels
                concat_flag = False
            else:
                out_dim_conv = hidden_channels
                concat_flag = True

            self.layer_list.append(
                TransformerConv(
                    in_channels=current_x_dim,
                    out_channels=hidden_channels if is_last_layer else self.head_dim,
                    heads=attention_heads,
                    concat=concat_flag
                )
            )
            
            self.norm_list.append(torch.nn.LayerNorm(out_dim_conv))

            mlp_in = current_gn_dim + out_dim_conv
            mlp_out = out_dim_conv 

            self.gn_mlp_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=mlp_in, out_features=hidden_channels),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(in_features=hidden_channels, out_features=mlp_out)
                )
            )
            
            self.gn_norm_list.append(torch.nn.LayerNorm(mlp_out))

            current_x_dim = out_dim_conv
            current_gn_dim = out_dim_conv

        self.output_layer = torch.nn.Linear(
            in_features=current_x_dim,
            out_features=classes
        )

    def forward(self, x, edge_index):
        current_gn_state = self.initial_gn_state

        for layer, gn_mlp, norm, gn_norm in zip(self.layer_list, self.gn_mlp_list, self.norm_list, self.gn_norm_list):
            
            x = x + current_gn_state
            
            x = F.dropout(x, p=0.5, training=self.training)
            x = layer(x, edge_index)
            
            x = norm(x)
            x = self.leaky_relu(x)
            
            global_pool = torch.mean(input=x, dim=0, keepdim=True)
            current_gn_state = gn_mlp(torch.cat([current_gn_state, global_pool], dim=1))
            current_gn_state = gn_norm(current_gn_state)
            
        x = x + current_gn_state
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.output_layer(x)
        
        return x

