from torch_geometric.nn import TransformerConv
import torch
from data_loader import get_datasets

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
        self.softmax = torch.nn.Softmax(dim=1)
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
        return self.softmax(x)