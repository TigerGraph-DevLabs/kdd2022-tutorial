class GAT(torch.nn.Module):
    def __init__(
        self, num_layers, out_dim, dropout, hidden_dim, num_heads
    ):
        super().__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_units = (-1, -1) if i == 0 else hidden_dim * num_heads
            out_units = out_dim if i == (num_layers - 1) else hidden_dim
            heads = 1 if i == (num_layers - 1) else num_heads
            self.layers.append(
                GATConv(in_units, out_units, heads=heads, dropout=dropout)
            )
        self.double()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x