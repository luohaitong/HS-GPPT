import torch
import math
import numpy as np
import random
import os
import pickle
from torch_geometric.data import Dataset
import scipy.sparse as sp
import dgl
from dataset_loader import DataLoader
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
# from cSBM import dataset_loader
# from .cSBM.dataset_loader import dataset_ContextualSBM
from torch_geometric.data import InMemoryDataset
from datetime import datetime
import os.path as osp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    num_nodes = label.shape[0]
    index=[i for i in range(num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(label.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            # train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
            train_idx.extend(rnd_state.choice(class_idx, 0, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=num_nodes)
    val_mask = index_to_mask(val_idx,size=num_nodes)
    test_mask = index_to_mask(test_idx,size=num_nodes)
    return train_mask, val_mask, test_mask

def load_induced_graph(dataset_name, device):

    folder_path = '/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/induced_graph/' + dataset_name
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = folder_path + '/induced_graph_min100_max300.pkl'
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list

class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        初始化 GraphDataset
        :param graphs: 包含图对象的列表
        """
        super(GraphDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        """
        返回数据集的大小
        :return: 数据集的大小
        """
        return len(self.graphs)

    def get(self, idx):
        """
        获取索引为 idx 的图
        :param idx: 索引
        :return: 图对象
        """
        graph = self.graphs[idx]
        # 可以在这里进行图数据的预处理或特征提取
        # 例如，如果每个图对象都有节点特征和边特征，可以返回它们
        # return {'node_features': graph.node_features, 'edge_index': graph.edge_index}
        return graph

# def transform_graph(x, edge_index):
#     edge_index = edge_index.T
#     adj = sp.coo_matrix((np.ones(len(edge_index)), (edge_index.cpu()[:, 1], edge_index.cpu()[:, 0])),
#                         shape=(len(x), len(x)))
#     graph = dgl.from_scipy(adj)
#     graph = dgl.to_bidirected(graph)
#     graph = dgl.add_self_loop(graph)

#     return graph

def transform_graph(x, edge_index):
    # 确保edge_index和x是PyTorch张量并且requires_grad=True
    edge_index = edge_index.T
    edge_index = edge_index.to(torch.long)  # 确保edge_index是整数类型的张量
    x = x.to(torch.float)  # 确保x是浮点类型的张量

    # 创建DGL图
    graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.size(0))

    # 将节点特征x添加到图中
    graph.ndata['feat'] = x

    # 将图移动到CPU
    graph = graph.to('cpu')
    # 将图转换为双向图并添加自环
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)



    # # 转换为简单图
    # graph = dgl.to_simple(graph)

    return graph

class dataset_ContextualSBM(InMemoryDataset):
    r"""Create synthetic dataset based on the contextual SBM from the paper:
    https://arxiv.org/pdf/1807.09596.pdf

    Use the similar class as InMemoryDataset, but not requiring the root folder.

       See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
    tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset if not specified use time stamp.

        for {n, d, p, Lambda, mu}, with '_' as prefix: intial/feed in argument.
        without '_' as prefix: loaded from data information

        n: number nodes
        d: avg degree of nodes
        p: dimenstion of feature vector.

        Lambda, mu: parameters balancing the mixture of information,
                    if not specified, use parameterized method to generate.

        epsilon, theta: gap between boundary and chosen ellipsoid. theta is
                        angle of between the selected parameter and x-axis.
                        choosen between [0, 1] => 0 = 0, 1 = pi/2

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    #     url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root, name=None,
                 n=800, d=5, p=100, Lambda=None, mu=None,
                 epsilon=0.1, theta=0.5,
                 # train_percent=0.01,
                 transform=None, pre_transform=None):

        now = datetime.now()
        surfix = now.strftime('%b_%d_%Y-%H:%M')
        if name is None:
            # not specifing the dataset name, create one with time stamp.
            self.name = '_'.join(['cSBM_data', surfix])
        else:
            self.name = name

        self._n = n
        self._d = d
        self._p = p

        self._Lambda = Lambda
        self._mu = mu
        self._epsilon = epsilon
        self._theta = theta

        # self._train_percent = train_percent

        root = osp.join(root, self.name)
        if not osp.isdir(root):
            os.makedirs(root)
        super(dataset_ContextualSBM, self).__init__(
            root, transform, pre_transform)

        #         ipdb.set_trace()

        self.data, self.slices = torch.load(self.processed_paths[0])
        # overwrite the dataset attribute n, p, d, Lambda, mu
        # self.Lambda = self.data.Lambda.item()
        self.mu = self.data.mu
        self.n = self.data.n
        self.p = self.data.p
        self.d = self.data.d
        # self.train_percent = self.data.train_percent

    #     @property
    #     def raw_dir(self):
    #         return osp.join(self.root, self.name, 'raw')

    #     @property
    #     def processed_dir(self):
    #         return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.raw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                # if self._Lambda is None or self._mu is None:
                #     # auto generate the lambda and mu parameter by angle theta.
                #     self._Lambda, self._mu = parameterized_Lambda_and_mu(self._theta,
                #                                                          self._p,
                #                                                          self._n,
                #                                                          self._epsilon)
                tmp_data = ContextualSBM(self._n,
                                         self._d,
                                         self._Lambda,
                                         self._p,
                                         self._mu)
                                # self._train_percent
                _ = save_data_to_pickle(tmp_data,
                                        p2root=self.raw_dir,
                                        file_name=self.name)
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
    
def load_data_cSBM(dataname, device, num_channels):

    path = './cSBM/data/'
    dataset = dataset_ContextualSBM(root=path, name=dataname)
    data = dataset[0]

    feat = data.x
    label = data.y
    edge_index = data.edge_index

    print(feat.shape)
    print(label.shape)
    print(edge_index.shape)

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    edge_index = edge_index.to(device)
    feat = feat.to(device)

    n_node = feat.shape[0]
    k = num_channels
    lbl1 = torch.ones(n_node * k)
    lbl2 = torch.zeros(n_node * k)
    lbl = torch.cat((lbl1, lbl2))

    return feat, edge_index, label, lbl, n_feat, n_classes


def load_data_multiple(dataname, device, num_channels):
    if dataname in ['amazon_ratings','roman_empire']:
        data = np.load(f'/home/dell/luohaitong/he_prompt/PolyGCL/data/{dataname}.npz')
        feat = torch.tensor(data['node_features'])
        label = torch.tensor(data['node_labels'])
        edge_index = torch.tensor(data['edges']).T

    else:
        dataset = DataLoader(name=dataname)
        if dataname in ['chameleon', 'actor', 'squirrel']:
            data = torch.load(f'./data/{dataname}/data.pt')
        else:
            data = dataset[0]

        feat = data.x
        label = data.y
        edge_index = data.edge_index

    print(feat.shape)
    print(label.shape)
    print(edge_index.shape)

    # pca = PCA(n_components=128)
    # feat = torch.FloatTensor(pca.fit_transform(feat))
    scaler = StandardScaler()
    svd = TruncatedSVD(n_components=128)
    # feat = torch.FloatTensor(scaler.fit_transform(feat))
    feat = torch.FloatTensor(svd.fit_transform(feat))

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    edge_index = edge_index.to(device)
    feat = feat.to(device)

    n_node = feat.shape[0]
    k = num_channels
    lbl1 = torch.ones(n_node * k)
    lbl2 = torch.zeros(n_node * k)
    lbl = torch.cat((lbl1, lbl2))

    return feat, edge_index, label.to(device), lbl, n_feat, n_classes

def load4graph(dataset_name, pretrained=False):
    r"""A plain old python object modeling a batch of graphs as one big
        (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
        base class, all its methods can also be used here.
        In addition, single graphs can be reconstructed via the assignment vector
        :obj:`batch`, which maps each node to its respective graph identifier.
        """

    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD']:
        dataset = TUDataset(root='/home/dell/luohaitong/he_prompt/ProG_new/ProG/data/TUDataset', name=dataset_name, use_node_attr=True)  # use_node_attr=False时，节点属性为one-hot编码的节点类别
        input_dim = dataset.num_features
        out_dim = dataset.num_classes
        torch.manual_seed(12345)

        if pretrained:
            dataset = dataset.shuffle()
        graph_list = []

        if 1:
            svd = TruncatedSVD(n_components=16)
            for data in dataset:
                data.x = torch.FloatTensor(svd.fit_transform(data.x))
                if data.x.shape[1] != 16:
                    continue
                graph_list.append(data)
            input_dim = 16 
        else:
            graph_list = [data for data in dataset]

        if dataset_name in ['COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY']:
            graph_list = [g for g in graph_list]
            node_degree_as_features(graph_list)
            input_dim = graph_list[0].x.size(1)       

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return input_dim, out_dim, graph_list  # 统一下游任务返回参数的顺序
        
    if dataset_name in ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']:
        dataset = PygGraphPropPredDataset(name = dataset_name, root='./dataset')
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        graph_list = [data for data in dataset]

        graph_list = [g for g in graph_list]
        # node_degree_as_features(graph_list)
        input_dim = graph_list[0].x.size(1)

        for g in graph_list:
            g.y = g.y.squeeze(0)

        if(pretrained==True):
            return input_dim, out_dim, graph_list
        else:
            return dataset, input_dim, out_dim
        
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import KBinsDiscretizer

def row_normalize_adj(adj):
    """Row-normalize feature matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def generate_adj(edge_index, num_items):
    edge_index = edge_index.T
    '''generate sparse adj'''
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_items, num_items), dtype=np.float32)

    #Turn the matrix into a symmetric matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_norm = row_normalize_adj(adj)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    print("Finish generating adjacent matrix")

    return adj_norm