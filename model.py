import torch
import scipy
import sympy
import dgl.function as fn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
import numpy as np

def calculate_theta(d):
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
        # Step 1: Obtain A
        A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        A[edge_index[0], edge_index[1]] = 1
        A[edge_index[1], edge_index[0]] = 1

        # Step 2: Diagonal D
        deg = A.sum(dim=1)
        D = torch.diag(deg)

        # Step 3: Obtain D^-1/2
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0

        # Step 4: Obtain L
        L = torch.eye(num_nodes, device=edge_index.device) - torch.matmul(D_inv_sqrt, torch.matmul(A, D_inv_sqrt))

        return L
    
    def forward(self, edge_index, feat):

        def unnLaplacian(feat, D_invsqrt, edge_index):
            """ Operation Feat - Feat * D^-1/2 A D^-1/2 """
            h = feat * D_invsqrt
            h = self.efficient_message_passing(h, edge_index)

            return feat - h * D_invsqrt

        row, col = edge_index
        deg = degree(row, feat.shape[0], dtype=feat.dtype)  # [N, ]
        
        D_invsqrt = torch.pow(deg.float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
        h = self._theta[0] * feat
        for k in range(1, self._k):
            # L_normalized * feat
            feat = unnLaplacian(feat, D_invsqrt, edge_index)
            h += self._theta[k] * feat

        h = self.linear(h)
        h = self.activation(h)

        return h
    
class BWGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d = 2):
        super(BWGNN, self).__init__()

        self.thetas = calculate_theta(d=d)
        self.conv = torch.nn.ModuleList()
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(in_channels, out_channels, self.thetas[i], lin=True))

    def forward(self, g, in_feat):

        h_final = []
        i = 0
        for conv in self.conv:
            h0 = conv(g, in_feat)
            h_final.append(h0)
            i += 1

        return h_final
    
    def get_embedding(self, g1, g2, g3, in_feat1, in_feat2, in_feat3):

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
        for h in h_neg_list:
            sc_list.append(self.fn(h, c_x).squeeze(1))

        logits = torch.cat(sc_list)

        return logits

class FilterWeight(torch.nn.Module):
    def __init__(self, feature_dim, num_channels=11):
        super(FilterWeight, self).__init__()
        self.num_channels = num_channels
        self.weight_channels = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(feature_dim) * (1.0), requires_grad=True) for _ in range(num_channels)])
        self.linear = torch.nn.Linear(feature_dim,feature_dim)
        self.act = torch.nn.ReLU()

    def forward(self, node_embeddings):
        
        sum = 0.0
        for i in range(self.num_channels):
            sum += torch.exp(self.weight_channels[i])
        for i in range(self.num_channels):
            if i == 0:
                weight_tem = torch.exp(self.weight_channels[i])/sum
                result = torch.mul(weight_tem, node_embeddings[i])
            else:
                weight_tem = torch.exp(self.weight_channels[i])/sum
                result += torch.mul(weight_tem, node_embeddings[i])
        return result
    
class LogReg(torch.nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = torch.nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret