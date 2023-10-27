import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# from torch.nn import GINConv, global_add_pool
# from torch.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.c=channel
        self.fc = nn.Sequential(
            nn.Linear(self.c, self.c // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c // reduction, self.c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        # torch.Size([128, 96, 128])
        y = self.avg_pool(x)
        # print(y.shape)  # [128, 96, 1]
        y=y.view(b, c)
        # print(y.shape) # [128, 96]
        y = self.fc(y)
        # print(y.shape) # [128, 96]
        y=y.view(b, c, 1)
        return x * y.expand_as(x)
class Dense_Block(nn.Module):
    def __init__(self, in_channels, n_filter, k, dropRate=0.0):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=k)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=n_filter, kernel_size=k)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=n_filter, kernel_size=k)
        self.SELayer = SELayer(channel=n_filter*3)
        self.dropRate=dropRate
    def forward(self, x):
        # bn = self.bn(x)
        conv1 = self.relu(self.conv1(x))
        conv1 = F.dropout(conv1, p=self.dropRate, training=self.training)
        # print(conv1.shape) #[128,32,123]
        conv2 = self.relu(self.conv2(conv1))


        # print(conv2.shape) #[128,32,118]

        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        conv3 = F.dropout(conv3, p=self.dropRate, training=self.training)
        # print(conv3.shape)
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
        # print(c3_dense.shape)
        out = self.SELayer(c3_dense)
        # print()
        # print(out.shape)
        return out


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
    def forward(self, x):
        # bn = self.bn(self.relu(self.conv(x)))
        bn = self.relu(self.conv(x))
        out = self.avg_pool(bn)
        return out
class DenseNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(DenseNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        # replace DenseNet
        self.denseblock1 = Dense_Block(in_channels=1000, n_filter=n_filters,k=1, dropRate=0.2)
        self.denseblock2 = Dense_Block(in_channels=96, n_filter=n_filters,k=1, dropRate=0.2)
        self.denseblock3 = Dense_Block(in_channels=96, n_filter=n_filters,k=1, dropRate=0.2)
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 96, out_channels = n_filters)
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 96, out_channels = n_filters)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*3*128, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

        # 加载预训练好的BERT模型和tokenizer
        # 普通bert
        # self.k_bert = BertModel.from_pretrained('bert-base-uncased')
        from transformers import DistilBertTokenizer, DistilBertModel
        # model_name = '/distilbert-base-uncased'
        model_name = 'huawei-noah/TinyBERT_General_4L_312D'
        # self.k_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # self.k_bert = BertModel.from_pretrained(model_name)
        #k-bert
        # model_path = './models/k_bert.pth'
        # self.k_bert = torch.load(model_path)
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, data):
        target = data.target

        # print(x.shape)
        embedded_xt = self.embedding_xt(target) #[num_feature+1,embed_dim] [79->128]
        # print('0', embedded_xt.shape)
        denseout = self.denseblock1(embedded_xt)
        denseout = self.denseblock2(denseout)
        denseout = self.denseblock3(denseout)


        # flatten
        xt = denseout.view(-1, 32 *3* 128)
        xt = self.fc1_xt(xt)
        xc=xt
        # add some dense layers

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)