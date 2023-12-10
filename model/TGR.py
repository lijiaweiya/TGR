import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import warnings

warnings.filterwarnings("ignore")


class TGR(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, layers, num_classes=3, dropout=0.4, pooling=0,
                 conv='GATv2'):
        """
        :param in_channels:     输入特征维度
        :param hidden_channels: 隐藏层特征维度
        :param num_heads:       GAT多头注意力的头数
        :param layers:          GAT层数
        :param num_classes:     输出类别数
        :param dropout:         dropout概率
        :param pooling:         0:mean_pooling+max_pooling,1:mean_pooling,2:max_pooling
        :param conv:            卷积类型
        """
        super(TGR, self).__init__()
        if conv == 'GATv2':
            self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, heads=num_heads,
                                   dropout=dropout,
                                   add_self_loops=True
                                   )
            self.conv2 = torch.nn.ModuleList([GATv2Conv(in_channels=hidden_channels * num_heads,
                                                        out_channels=hidden_channels, heads=num_heads, dropout=dropout,
                                                        add_self_loops=True
                                                        ) for
                                              i in range(layers - 1)])
        elif conv == 'GAT':
            self.conv1 = GATConv(in_channels=in_channels, out_channels=hidden_channels, heads=num_heads, dropout=dropout)
            self.conv2 = torch.nn.ModuleList([GATConv(in_channels=hidden_channels * num_heads,
                                                       out_channels=hidden_channels, heads=num_heads, dropout=dropout)
                                              for i in range(layers - 1)])
        elif conv == 'GCN':
            self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels* num_heads)
            self.conv2 = torch.nn.ModuleList([GCNConv(in_channels=hidden_channels* num_heads,
                                                       out_channels=hidden_channels* num_heads)
                                              for i in range(layers - 1)])
        self.lin1 = Linear(hidden_channels * num_heads, num_classes)
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, edge_index, batch):
        #print(x.shape, edge_index.shape, batch.shape)
        x = self.conv1(x, edge_index)
        for m in self.conv2:
            F.relu(x, inplace=True)
            x = m(x, edge_index)

        match self.pooling:
            case 0:
                x = global_mean_pool(x, batch) \
                    + global_max_pool(x, batch)
            case 1:
                x = global_mean_pool(x, batch)
            case 2:
                x = global_max_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.softmax(x, dim=1)
        return x
