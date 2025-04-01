import torch
from torch_geometric.data import Batch, Data
    
class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, mean, std, group_num=1, inner_prune=None):

        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune
        
        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])
        
        self.token_init(init_method="kaiming_uniform")
        
        self.mean = mean
        self.std = std

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):

        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token

            if self.mean is not None:
                mean_p = torch.mean(tokens, dim=0)
                std_p = torch.std(tokens, dim=0)
                std_p = torch.clamp(std_p, min=1e-6)

                tokens = (tokens-mean_p) / std_p
                tokens = tokens * self.std + self.mean

            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch

class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, mean, std, cross_prune=0.5, inner_prune=0.3):
        super(HeavyPrompt, self).__init__(token_dim, token_num, mean, std, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def add(self, x, edge_index, cross_prune_init=None):

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        new_edge_index = edge_index + token_num
        
        cross_dot = torch.mm(pg.x, torch.transpose(x, 0, 1))
        cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
        if cross_prune_init is not None:
            cross_adj = torch.where(cross_sim < cross_prune_init, 0, cross_sim)
        else:
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
        
        cross_edge_index = cross_adj.nonzero().t().contiguous()
        cross_edge_index[1] = cross_edge_index[1] + token_num
        
        new_x = torch.cat([pg.x, x], dim=0)

        edge_index = torch.cat([inner_edge_index, new_edge_index, cross_edge_index], dim=1)
        edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)],dim=1), dim=1)

        return edge_index, new_x