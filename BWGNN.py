import torch
import scipy
import sympy
import dgl.function as fn
import torch.nn.functional as F

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
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            torch.nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = (feat * D_invsqrt)
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                # L_normalized * feat
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
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
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(out_channels, out_channels, self.thetas[i], lin=False))
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.d = d
        # self.weight_channels = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0/3.0), requires_grad=True) for _ in range(len(self.conv))])
        self.weight_channels = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(out_channels) * (1.0), requires_grad=True) for _ in range(len(self.conv))])
        self.disc = Discriminator(out_channels)

    def get_embedding(self, g, in_feat1, in_feat2, in_feat3):

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
                h0 = conv(g, in_feat1)
            elif i==1:
                h0 = conv(g, in_feat2)
            elif i==2:
                h0 = conv(g, in_feat3)
            else:
                h0 = conv(g, in_feat1)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            h0 = self.linear(h0)
            h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            h_final.append(h0)
            i += 1

        # h_final = []
        # i = 0
        # for conv in self.conv:
        #     if i==0:
        #         h0 = conv(g, in_feat1)
        #     elif i==1:
        #         h0 = conv(g, in_feat2)
        #     elif i==2:
        #         h0 = conv(g, in_feat3)
        #     # h0 = conv(g, in_feat1)
        #     # h0 = self.linear(h0)
        #     # h0 = self.act(h0)
        #     # h0 = self.linear2(h0)
        #     h_final.append(h0)
        #     i+=1

        # for i in range(len(self.conv)):
        #     if i == 0:
        #         h_final_pos = torch.mul(self.weight_channels[i], h_final[i])
        #     else:
        #         h_final_pos += torch.mul(self.weight_channels[i], h_final[i])
        # h_final_pos = torch.cat(h_final, dim=-1)

        return h_final
    
    def forward(self, g, in_feat, shuffle_in_feat):

        # if self.dprate != 0.0:
        #     in_feat = F.dropout(in_feat, p=self.dprate, training=self.training)
        #     shuffle_in_feat = F.dropout(shuffle_in_feat, p=self.dprate, training=self.training)
    
        h_final = []
        for conv in self.conv:
            h0 = conv(g, in_feat)
            # h0 = F.dropout(h0, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h0 = self.bn(h0)
            h0 = self.linear(h0)
            h0 = self.act_fn(h0)
            # h0 = self.linear2(h0)
            h_final.append(h0)

        # for i in range(len(self.conv)):
        #     if i == 0:
        #         h_final_pos = torch.mul(self.weight_channels[i], h_final[i])
        #     else:
        #         h_final_pos += torch.mul(self.weight_channels[i], h_final[i])
        # for i in range(len(self.conv)):
        #     if i == 0:
        #         h_final_pos = torch.mul(self.weight_channels[i], h_final[i])
        #     elif i == 1:
        #         h_final_pos += torch.mul(self.weight_channels[i], h_final[i])
        #     elif i == 2:
        #         h_final_pos += torch.mul(1-self.weight_channels[0]-self.weight_channels[1], h_final[i])
        for i in range(len(self.conv)):
            if i == 0:
                weight_tem = torch.exp(self.weight_channels[i])/(torch.exp(self.weight_channels[0])+torch.exp(self.weight_channels[1])+torch.exp(self.weight_channels[2]))
                h_final_pos = torch.mul(weight_tem, h_final[i])
            else:
                weight_tem = torch.exp(self.weight_channels[i])/(torch.exp(self.weight_channels[0])+torch.exp(self.weight_channels[1])+torch.exp(self.weight_channels[2]))
                h_final_pos += torch.mul(weight_tem, h_final[i])
            

        h_final_shuffle = []
        for conv in self.conv:
            h1 = conv(g, shuffle_in_feat)
            # h1 = F.dropout(h1, p=self.dropout, training=self.training)
            # if self.is_bns:
            #     h1 = self.bn(h1)
            h1 = self.linear(h1)
            h1 = self.act_fn(h1)
            # h1 = self.linear2(h1)
            h_final_shuffle.append(h1)
        
        c = self.act(torch.mean(h_final_pos, dim=0))

        out = self.disc(h_final, h_final_shuffle, c)


        return out
    
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
        # for h in h_pos_list:
        #     # tem = torch.sum(torch.mul(h, c_x),dim=1)
        #     tem = torch.cosine_similarity(h, c_x, dim=1)
        #     sc_list.append(tem)
        # for h in h_neg_list:
        #     #tem = torch.sum(torch.mul(h, c_x),dim=1)
        #     tem = torch.cosine_similarity(h, c_x, dim=1)
        #     sc_list.append(tem)

        logits = torch.cat(sc_list)

        return logits