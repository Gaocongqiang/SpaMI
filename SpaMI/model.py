import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class SpaMI(nn.Module):
    def __init__(self, omics1_input_dim, omics2_input_dim, out_dim, dropout_rate,
                 omics1_feat, omics1_adj, omics1_graph_neigh, omics1_label_CSL,
                 omics2_feat, omics2_adj, omics2_graph_neigh, omics2_label_CSL
                 ):
        super(SpaMI, self).__init__()

        # dimensions of input and output
        self.omics1_input_dim = omics1_input_dim
        self.omics2_input_dim = omics2_input_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        # omics1 data
        self.omics1_feat = omics1_feat
        self.omics1_adj = omics1_adj
        self.omics1_graph_neigh = omics1_graph_neigh
        self.omics1_label_CSL = omics1_label_CSL

        # omics2 data
        self.omics2_feat = omics2_feat
        self.omics2_adj = omics2_adj
        self.omics2_graph_neigh = omics2_graph_neigh
        self.omics2_label_CSL = omics2_label_CSL

        # encoder
        self.omics1_encoder = omics1_encoder(self.omics1_input_dim, self.out_dim, self.dropout_rate)
        self.omics2_encoder = omics2_encoder(self.omics2_input_dim, self.out_dim, self.dropout_rate)

        # decoder
        self.omics1_decoder = omics1_decoder(self.omics1_input_dim, self.out_dim, self.dropout_rate)
        self.omics2_decoder = omics2_decoder(self.omics2_input_dim, self.out_dim, self.dropout_rate)

        # attention aggregation layer
        self.attention_layer = AttentionLayer(out_dim, out_dim)

    def forward(self, omics1_feat_shuffle, omics2_feat_shuffle):
        omics1_emb, omics1_ret, omics1_ret_a = self.omics1_encoder(self.omics1_feat, omics1_feat_shuffle,
                                                                   self.omics1_adj, self.omics1_graph_neigh)

        omics2_emb, omics2_ret, omics2_ret_a = self.omics2_encoder(self.omics2_feat, omics2_feat_shuffle,
                                                                   self.omics2_adj, self.omics2_graph_neigh)

        combine_emb, omics_weight = self.attention_layer(omics1_emb, omics2_emb)

        omics1_rec = self.omics1_decoder(combine_emb, self.omics1_adj)
        omics2_rec = self.omics2_decoder(combine_emb, self.omics2_adj)

        return omics1_emb, omics1_rec, omics1_ret, omics1_ret_a, omics2_emb, omics2_rec, omics2_ret, omics2_ret_a, combine_emb, omics_weight


class omics1_encoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics1_encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_conv1 = GCNConv(input_dim, 256)
        self.encoder_conv2 = GCNConv(256, out_dim)

        self.discriminator = Discriminator(out_dim)
        self.readout = AvgReadout()

    def forward(self, feat, feat_a, adj, graph_neigh):
        edge_index, _ = dense_to_sparse(adj)

        z = self.dropout(feat)
        z = self.encoder_conv1(z, edge_index)
        z = F.relu(z)
        z = self.encoder_conv2(z, edge_index)
        emb = z

        z_a = self.dropout(feat_a)
        z_a = self.encoder_conv1(z_a, edge_index)
        z_a = F.relu(z_a)
        z_a = self.encoder_conv2(z_a, edge_index)
        emb_a = z_a

        # Local neighbor representation
        g = self.readout(emb, graph_neigh)
        g = F.sigmoid(g)

        # Corrupted local neighbor representation
        g_a = self.readout(emb_a, graph_neigh)
        g_a = F.sigmoid(g_a)

        # discriminator for contrastive learning
        ret = self.discriminator(g, emb, emb_a)
        ret_a = self.discriminator(g_a, emb_a, emb)

        return emb, ret, ret_a


class omics1_decoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics1_decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_conv1 = GCNConv(out_dim, 256)
        self.decoder_conv2 = GCNConv(256, input_dim)

    def forward(self, emb, adj):
        edge_index, _ = dense_to_sparse(adj)

        h = self.dropout(emb)
        h = self.decoder_conv1(h, edge_index)
        h = F.relu(h)
        h = self.decoder_conv2(h, edge_index)
        omics1_rec = h

        return omics1_rec


class omics2_encoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics2_encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_conv1 = GCNConv(input_dim, 256)
        self.encoder_conv2 = GCNConv(256, out_dim)

        self.discriminator = Discriminator(out_dim)
        self.readout = AvgReadout()

    def forward(self, feat, feat_a, adj, graph_neigh):
        edge_index, _ = dense_to_sparse(adj)

        z = self.dropout(feat)
        z = self.encoder_conv1(z, edge_index)
        z = F.relu(z)
        z = self.encoder_conv2(z, edge_index)
        emb = z

        z_a = self.dropout(feat_a)
        z_a = self.encoder_conv1(z_a, edge_index)
        z_a = F.relu(z_a)
        z_a = self.encoder_conv2(z_a, edge_index)
        emb_a = z_a

        # Local neighbor representation
        g = self.readout(emb, graph_neigh)
        g = F.sigmoid(g)

        # Corrupted local neighbor representation
        g_a = self.readout(emb_a, graph_neigh)
        g_a = F.sigmoid(g_a)

        ret = self.discriminator(g, emb, emb_a)
        ret_a = self.discriminator(g_a, emb_a, emb)

        return emb, ret, ret_a


class omics2_decoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics2_decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_conv1 = GCNConv(out_dim, 256)
        self.decoder_conv2 = GCNConv(256, input_dim)

    def forward(self, emb, adj):
        edge_index, _ = dense_to_sparse(adj)

        h = self.dropout(emb)
        h = self.decoder_conv1(h, edge_index)
        h = F.relu(h)
        h = self.decoder_conv2(h, edge_index)
        omics2_rec = h

        return omics2_rec


class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.w_omega = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = nn.Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = torch.cat((torch.unsqueeze(emb1, 1), torch.unsqueeze(emb2, 1)), 1)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=1)
        emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))
        return torch.squeeze(emb_combined), alpha


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
