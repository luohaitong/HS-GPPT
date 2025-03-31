import torch
import scipy
import sympy
import dgl.function as fn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool
import numpy as np
from GNN_baseline import GCN_Low2,GCN_Mid2,GCN_High2

def get_first_occurrence_indices(tensor):
    unique_values = np.unique(tensor)
    first_occurrence_indices = []
    for value in unique_values:
        indices = np.where(tensor == value)[0]
        first_occurrence_indices.append(indices[0])
    return first_occurrence_indices

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class PolyConv(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = torch.nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        self.linear2 = torch.nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    # def forward(self, graph, feat):
    #     def unnLaplacian(feat, D_invsqrt, graph):
    #         """ Operation Feat * D^-1/2 A D^-1/2 """
    #         graph.ndata['h'] = (feat * D_invsqrt)
    #         graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    #         return feat - graph.ndata.pop('h') * D_invsqrt

    #     with graph.local_scope():
    #         D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
    #             min=1), -0.5).unsqueeze(-1).to(feat.device)
    #         h = self._theta[0]*feat
    #         for k in range(1, self._k):
    #             # L_normalized * feat
    #             feat = unnLaplacian(feat, D_invsqrt, graph)
    #             h += self._theta[k]*feat
    #     if self.lin:
    #         h = self.linear(h)
    #         h = self.activation(h)
    #     return h

    def efficient_message_passing(self, feat, edge_index):

        num_nodes = feat.size(0)
        src, dst = edge_index
        out = torch.zeros((num_nodes, feat.size(1)), device=feat.device)
        out.scatter_add_(0, dst.unsqueeze(-1).repeat(1,feat.shape[1]), feat[src])
        
        return out
    
    def message_passing(self, feat, edge_index):

        num_nodes = feat.size(0)
        src, dst = edge_index
        out = torch.zeros((num_nodes, feat.size(1)), device=feat.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            out[dst] += feat[src]
        
        return out

    def normalized_laplacian(self, edge_index, num_nodes):
        # Step 1: 构建邻接矩阵A
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1
        A[edge_index[1], edge_index[0]] = 1  # 无向图，添加反向边

        # Step 2: 计算度矩阵D
        deg = A.sum(dim=1)
        D = torch.diag(deg)

        # Step 3: 计算D的逆平方根
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0  # 避免除以零

        # Step 4: 计算归一化拉普拉斯矩阵L
        L = torch.eye(num_nodes, device=edge_index.device) - torch.matmul(D_inv_sqrt, torch.matmul(A, D_inv_sqrt))

        return L
    
    # def forward(self, edge_index, feat):

    #     L_normalized = self.normalized_laplacian(edge_index, len(feat))
    #     h = self._theta[0] * feat
    #     for k in range(1, self._k):
    #         # L_normalized * feat
    #         feat = torch.matmul(L_normalized, feat)
    #         h += self._theta[k] * feat

    #     if self.lin:
    #         h = self.linear(h)
    #         h = self.activation(h)
    #     return h

    def forward(self, edge_index, feat):

        def unnLaplacian(feat, D_invsqrt, edge_index):
            """ Operation Feat - Feat * D^-1/2 A D^-1/2 """
            h = feat * D_invsqrt
            h = self.efficient_message_passing(h, edge_index)

            return feat - h * D_invsqrt

        row, col = edge_index
        deg = degree(row, feat.shape[0], dtype=feat.dtype)  # [N, ]
        
        # D_invsqrt = deg.pow(-0.5).unsqueeze(-1)  # [N, ]
        D_invsqrt = torch.pow(deg.float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
        h = self._theta[0] * feat
        for k in range(1, self._k):
            # L_normalized * feat
            feat = unnLaplacian(feat, D_invsqrt, edge_index)
            h += self._theta[k] * feat

        # if self.lin:
        #     h = self.linear(h)
        #     h = self.activation(h)
        # self.linear.to(h.device)
        # self.linear2.to(h.device)
        h = self.linear(h)
        h = self.activation(h)
        # h = self.linear2(h)
        return h
    
class BWGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dprate, dropout, is_bns, act_fn, d = 2):
        super(BWGNN, self).__init__()

        assert act_fn in ['relu', 'prelu']
        # self.act_fn = torch.nn.PReLU() if act_fn == 'prelu' else torch.nn.ReLU()
        self.act_fn = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_channels, momentum=0.01)
        self.dropout = dropout
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout

        self.thetas = calculate_theta2(d=d)
        # self.conv = []
        self.conv = torch.nn.ModuleList()
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(out_channels, out_channels, self.thetas[i], lin=True))
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.d = d
        self.pool = global_mean_pool

    def forward(self, g, in_feat):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        i = 0
        for conv in self.conv:
            # if i==0:
            #     h0 = conv(g, in_feat1)
            # elif i==1:
            #     h0 = conv(g, in_feat2)
            # elif i==2:
            #     h0 = conv(g, in_feat3)
            # else:
            #     h0 = conv(g, in_feat1)
            h0 = conv(g, in_feat)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)

            # h0 = self.linear(h0)
            # h0 = self.act_fn(h0)

            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1


        return h_final
    
    def forward_subgraph(self, edge_index, in_feat, batch):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        global_h_final = []
        i = 0
        for conv in self.conv:
            # if i==0:
            #     h0 = conv(g, in_feat1)
            # elif i==1:
            #     h0 = conv(g, in_feat2)
            # elif i==2:
            #     h0 = conv(g, in_feat3)
            # else:
            #     h0 = conv(g, in_feat1)
            h0 = conv(edge_index, in_feat)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            h0 = self.linear(h0)
            h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            global_h0 = self.pool(h0, batch.long())
            repeated_global_h0 = global_h0[batch.long()]
            h_final.append(h0)
            global_h_final.append(repeated_global_h0)
            i += 1


        return h_final, global_h_final
    
    def get_embedding(self, g1, g2, g3, in_feat1, in_feat2, in_feat3):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        i = 0
        for conv in self.conv:
            if i==0:
                h0 = conv(g1, in_feat1)
            elif i==1:
                h0 = conv(g2, in_feat2)
            elif i==2:
                h0 = conv(g3, in_feat3)
            else:
                h0 = conv(g1, in_feat1)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            # h0 = self.linear(h0)
            # h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1


        return h_final

    # def get_embedding(self, g1, g2, g3, g4, g5, in_feat1, in_feat2, in_feat3, in_feat4, in_feat5):

    #     # h_final = []
    #     # for conv in self.conv:
    #     #     h0 = conv(g, in_feat1)
    #     #     h0 = self.linear(h0)
    #     #     h0 = self.act(h0)
    #     #     h0 = self.linear2(h0)
    #     #     h_final.append(h0)

    #     # if self.dprate != 0.0:
    #     #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

    #     h_final = []
    #     i = 0
    #     for conv in self.conv:
    #         if i==0:
    #             h0 = conv(g1, in_feat1)
    #         elif i==1:
    #             h0 = conv(g2, in_feat2)
    #         elif i==2:
    #             h0 = conv(g3, in_feat3)
    #         elif i==3:
    #             h0 = conv(g4, in_feat4)
    #         elif i==4:
    #             h0 = conv(g5, in_feat5)
    #         else:
    #             raise ValueError
    #         # h0 = F.dropout(h0, p=self.dropout, training=self.training)
    #         # if self.is_bns:
    #         #     h0 = self.bn(h0)
    #         # h0 = self.linear(h0)
    #         # h0 = self.act_fn(h0)
    #         # h0 = self.linear2(h0)
    #         h_final.append(h0)
    #         i += 1


    #     return h_final

    # def get_embedding(self, g1, g2, g3, g4, in_feat1, in_feat2, in_feat3, in_feat4):

    #     # h_final = []
    #     # for conv in self.conv:
    #     #     h0 = conv(g, in_feat1)
    #     #     h0 = self.linear(h0)
    #     #     h0 = self.act(h0)
    #     #     h0 = self.linear2(h0)
    #     #     h_final.append(h0)

    #     # if self.dprate != 0.0:
    #     #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

    #     h_final = []
    #     i = 0
    #     for conv in self.conv:
    #         if i==0:
    #             h0 = conv(g1, in_feat1)
    #         elif i==1:
    #             h0 = conv(g2, in_feat2)
    #         elif i==2:
    #             h0 = conv(g3, in_feat3)
    #         elif i==3:
    #             h0 = conv(g4, in_feat4)
    #         else:
    #             raise ValueError
    #         # h0 = F.dropout(h0, p=self.dropout, training=self.training)
    #         # if self.is_bns:
    #         #     h0 = self.bn(h0)
    #         # h0 = self.linear(h0)
    #         # h0 = self.act_fn(h0)
    #         # h0 = self.linear2(h0)
    #         h_final.append(h0)
    #         i += 1


    #     return h_final
    # def get_embedding(self, g1, g2, in_feat1, in_feat2):

    #     # h_final = []
    #     # for conv in self.conv:
    #     #     h0 = conv(g, in_feat1)
    #     #     h0 = self.linear(h0)
    #     #     h0 = self.act(h0)
    #     #     h0 = self.linear2(h0)
    #     #     h_final.append(h0)

    #     # if self.dprate != 0.0:
    #     #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

    #     h_final = []
    #     i = 0
    #     for conv in self.conv:
    #         if i==0:
    #             h0 = conv(g1, in_feat1)
    #         elif i==1:
    #             h0 = conv(g2, in_feat2)
    #         else:
    #             raise ValueError

    #         h_final.append(h0)
    #         i += 1


    #     return h_final
    
    def get_embedding_subraph(self, graph_1, graph_2, graph_3):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        i = 0
        for conv in self.conv:
            if i==0:
                h0 = conv(graph_1.edge_index, graph_1.x)
                h0 = self.pool(h0, graph_1.batch.long())
                # indices = torch.tensor(get_first_occurrence_indices(graph_1.batch.cpu())).cuda()
                # h0 = h0[indices]
            elif i==1:
                h0 = conv(graph_2.edge_index, graph_2.x)
                h0 = self.pool(h0, graph_2.batch.long())
                # indices = torch.tensor(get_first_occurrence_indices(graph_2.batch.cpu())).cuda()
                # h0 = h0[indices]
            elif i==2:
                h0 = conv(graph_3.edge_index, graph_3.x)
                h0 = self.pool(h0, graph_3.batch.long())
                # indices = torch.tensor(get_first_occurrence_indices(graph_3.batch.cpu())).cuda()
                # h0 = h0[indices]
            else:
                h0 = conv(graph_1.edge_index, graph_1.x)
                h0 = self.pool(h0, graph_1.batch.long())
                # indices = torch.tensor(get_first_occurrence_indices(graph_1.batch.cpu())).cuda()
                # h0 = h0[indices]
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            h0 = self.linear(h0)
            h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1

        return h_final
    
class Triple_GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dprate, dropout, is_bns, act_fn, d = 2):
        super(Triple_GNN, self).__init__()

        assert act_fn in ['relu', 'prelu']
        # self.act_fn = torch.nn.PReLU() if act_fn == 'prelu' else torch.nn.ReLU()
        self.act_fn = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(in_channels, momentum=0.01)
        self.dropout = dropout
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout

        # self.thetas = calculate_theta2(d=d)
        # self.conv = []
        self.conv = torch.nn.ModuleList()
        # for i in range(len(self.thetas)):
        #     self.conv.append(PolyConv(out_channels, out_channels, self.thetas[i], lin=False))
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        # self.act = torch.nn.ReLU()
        self.act = torch.nn.LeakyReLU()
        self.d = d
        self.pool = global_mean_pool

        self.conv.append(GCN_Low2(in_channels, out_channels))
        self.conv.append(GCN_Mid2(in_channels, out_channels))
        self.conv.append(GCN_High2(in_channels, out_channels))



    def forward(self, g, in_feat):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        i = 0
        for conv in self.conv:
            # if i==0:
            #     h0 = conv(g, in_feat1)
            # elif i==1:
            #     h0 = conv(g, in_feat2)
            # elif i==2:
            #     h0 = conv(g, in_feat3)
            # else:
            #     h0 = conv(g, in_feat1)
            h0 = conv(in_feat, g)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)

            # h0 = self.linear(h0)
            # h0 = self.act_fn(h0)

            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1


        return h_final
    

    def get_embedding(self, g1, g2, g3, in_feat1, in_feat2, in_feat3):

        # h_final = []
        # for conv in self.conv:
        #     h0 = conv(g, in_feat1)
        #     h0 = self.linear(h0)
        #     h0 = self.act(h0)
        #     h0 = self.linear2(h0)
        #     h_final.append(h0)

        # if self.dprate != 0.0:
        #     in_feat1 = F.dropout(in_feat1, p=self.dprate, training=self.training)

        h_final = []
        i = 0
        for conv in self.conv:
            if i==0:
                h0 = conv(in_feat1, g1)
            elif i==1:
                h0 = conv(in_feat2, g2)
            elif i==2:
                h0 = conv(in_feat3, g3)
            else:
                h0 = conv(in_feat1, g1)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            h0 = self.linear(h0)
            h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1


        return h_final

class Discriminator(torch.nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = torch.nn.Bilinear(dim, dim, 1)

    def forward(self, h_pos_list, h_neg_list, c):
        
        c_x = c.expand_as(h_pos_list[0]).contiguous()

        sc_list = []
        for h in h_pos_list:
            sc_list.append(self.fn(h, c_x).squeeze(1))
            # sc_list.append(self.fn(h, c_x))
        for h in h_neg_list:
            sc_list.append(self.fn(h, c_x).squeeze(1))
            # sc_list.append(self.fn(h, c_x))

        logits = torch.cat(sc_list)

        return logits

    def forward_subgraph(self, h_pos_list, h_neg_list, c):
        
        # c_x = c.expand_as(h_pos_list[0]).contiguous()
        c_x = c
        sc_list = []
        for h in h_pos_list:
            sc_list.append(self.fn(h, c_x).squeeze(1))
        for h in h_neg_list:
            sc_list.append(self.fn(h, c_x).squeeze(1))

        logits = torch.cat(sc_list)

        return logits
    
    def prompt_tuning(self, embeds, label_embeds):
        # 扩展维度以进行批量的两两计算
        embeds_expanded = embeds.unsqueeze(1).repeat(1,label_embeds.shape[0],1)  # 形状变为 [N, 1, embed_size]
        label_embeds_expanded = label_embeds.unsqueeze(0).repeat(embeds_expanded.shape[0],1,1)  # 形状变为 [1, K, embed_size]

        # 进行批量的两两计算
        logits = self.fn(embeds_expanded, label_embeds_expanded).squeeze(-1)

        return logits

