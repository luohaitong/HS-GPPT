import torch
import numpy as np
import random
import scipy.sparse as sp
from dataset_loader import DataLoader
from sklearn.decomposition import TruncatedSVD



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


def load_data(dataname, device, num_channels):

    if dataname in ['amazon_ratings','roman_empire']:
        data = np.load(f'./data/{dataname}.npz')
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

    print("feature shape: ",feat.shape)
    print("edge index shape: ",edge_index.shape)

    svd = TruncatedSVD(n_components=128)
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
