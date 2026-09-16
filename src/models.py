from torch_geometric.nn import TransformerConv
import torch

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

        self.layer_list.append(
            TransformerConv(
                in_channels=input_channels,
                out_channels=hidden_channels,
                heads=attention_heads,
                concat=True
            )
        )
        for layer_number in range(1, layers):
            if layer_number == layers - 1:
                self.layer_list.append(
                    TransformerConv(
                        in_channels=hidden_channels * attention_heads,
                        out_channels=hidden_channels,
                        heads=attention_heads,
                        concat=False
                    )
                )

                break

            self.layer_list.append(
                TransformerConv(
                    in_channels=hidden_channels * attention_heads,
                    out_channels=hidden_channels,
                    heads=attention_heads,
                    concat=True
                )
            )

        self.output_layer = torch.nn.Linear(
                in_features=hidden_channels,
                out_features=classes
        )

        self.layers = torch.nn.ModuleList(self.layer_list)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.leaky_relu(x)

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
        self.layer_list = []
        self.gn_mlp_list = []
        self.last_layer = False
        self.transformer_input = input_channels
        self.mlp_input = attention_heads * hidden_channels + input_channels

        for layer in range(layers):
            if layer == layers - 1:
                self.last_layer = True
                self.mlp_input = hidden_channels + attention_heads * hidden_channels

            self.layer_list.append(
                TransformerConv(
                    in_channels=self.transformer_input,
                    out_channels=hidden_channels,
                    heads=attention_heads,
                    concat=not(self.last_layer)
                )
            )
            self.transformer_input = attention_heads * hidden_channels

            self.gn_mlp_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=self.mlp_input,
                        out_features = hidden_channels if self.last_layer else self.transformer_input
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(
                        in_features = hidden_channels if self.last_layer else self.transformer_input,
                        out_features = hidden_channels if self.last_layer else self.transformer_input
                    )
                )
            )
            self.mlp_input = (attention_heads * hidden_channels) * 2

        self.layers = torch.nn.ModuleList(self.layer_list)
        self.gn_mlps = torch.nn.ModuleList(self.gn_mlp_list)
        self.output_layer = torch.nn.Linear(
            in_features=hidden_channels,
            out_features=classes
        )

    def forward(self, x, edge_index):
        current_gn_state = self.initial_gn_state

        for layer, gn_mlp in zip(self.layers, self.gn_mlps):
            x += current_gn_state
            x = layer(x, edge_index)
            global_pool = torch.mean(
                input=x,
                dim=0,
                keepdim=True
            )
            current_gn_state = gn_mlp(torch.concat([current_gn_state, global_pool], dim=1))
        x += current_gn_state
        x = self.output_layer(x)
        return x