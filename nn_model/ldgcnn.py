from torch import nn
from torch.nn import functional
from torch_geometric import nn as gnn


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class SimpleClsLDGCN(nn.Module):

    def __init__(self, in_chan: int, out_chan: int, num_neighborhoods: int) -> None:
        assert num_neighborhoods // 2 > 1
        super().__init__()
        out_conv1_chan = 32
        self.conv1 = gnn.DynamicEdgeConv(MLP((2 * in_chan, 16, 32, out_conv1_chan)), k=num_neighborhoods)
        out_conv2_chan = 128
        self.conv2 = gnn.DynamicEdgeConv(MLP((2 * out_conv1_chan, 64, out_conv2_chan)), k=num_neighborhoods // 2)
        out_mlp_chan = 256
        self.mlp = MLP([out_conv2_chan, out_conv2_chan, out_mlp_chan])
        self.cls_layer = nn.Linear(out_mlp_chan, out_chan)

    def forward(self, data):
        global_features = self.global_feature(data)
        return functional.log_softmax(self.cls_layer(global_features), dim=1)

    def global_feature(self, data):
        pos, batch = data.pos, data.batch
        x = self.conv1(pos, batch)
        x = self.conv2(x, batch)
        global_features = gnn.global_max_pool(self.mlp(x), batch=batch)
        return global_features
