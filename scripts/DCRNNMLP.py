from tsl.nn.blocks import DCRNN, MultiHorizonMLPDecoder
import torch.nn as nn

class DCRNN_MLP(nn.Module):
    def __init__(self, input_size, static_size, exog_size, output_size, context_size, horizon):
        super(DCRNN_MLP, self).__init__()
        self.encoder = DCRNN(input_size,
                             hidden_size=64,
                             k=2
                             )
        self.decoder = MultiHorizonMLPDecoder(
            input_size=64,
            hidden_size=64,
            output_size=output_size,
            exog_size=exog_size,
            n_layers=2,
            context_size=context_size,
            horizon=horizon)

    def forward(self, x, u, edge_index, edge_weight):
        x, last = self.encoder(x, edge_index, edge_weight)
        out = self.decoder(x, u)
        return out

#exemple
"""model = DCRNN_MLP(
    input_size=input_size,
    static_size=static_size,
    exog_size=exog_size,
    output_size=1,
    context_size=64,
    horizon=24)"""