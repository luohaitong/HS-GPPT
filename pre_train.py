import argparse
import warnings
import torch
import random
import numpy as np
import os
from model import BWGNN, Discriminator, FilterWeight
from tqdm import *
from utils import load_data

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description="HS-GPPT")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--dataname", type=str, default="cora", help="Name of dataset.")
parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument( "--patience", type=int, default=100, help="Patient epochs to wait before early stopping.")
parser.add_argument("--lr", type=float, default=0.010, help="Learning rate.")
parser.add_argument("--lr1", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=0.0, help="Weight decay.")
parser.add_argument("--wd1", type=float, default=0.0, help="Weight decay.")
parser.add_argument("--hid_dim", type=int, default=128, help="Hidden layer dim.")
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu', help='activation function')
parser.add_argument('--num_filter', type=int, default=3, help='Number of hybrid spectral fitlers.')
args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

if __name__ == "__main__":
    print(args)
    # Step 1: Load data =================================================================== #

    feat, edge_index, label, lbl, n_feat, n_classes = load_data(args.dataname, args.device, num_channels=args.num_filter)
    lbl = lbl.to(args.device)

    # Step 2: Create model =================================================================== #
    model = BWGNN(n_feat, args.hid_dim, d = args.num_filter-1).to(args.device)
    filterweight = FilterWeight(args.hid_dim, num_channels=args.num_filter).to(args.device)
    disc = Discriminator(args.hid_dim).to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                        {'params': disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                        {'params': filterweight.parameters(), 'weight_decay': args.wd, 'lr': args.lr}
                        ])

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    best_t = 0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(feat.shape[0])
        shuf_feat = feat[shuf_idx, :]

        h1 = model(edge_index, feat)
        h1_shuffle = model(edge_index, shuf_feat)

        h_c = filterweight(h1)
        act = torch.nn.ReLU()
        c = act(torch.mean(h_c, dim=0))
        out = disc(h1, h1_shuffle, c)

        out_final = out
        lbl_final = lbl

        loss = loss_fn(out_final, lbl_final)

        loss.backward()
        optimizer.step()

        
        if epoch % 20 == 0:
            print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'pkl/best_model_'+ args.dataname +'.pkl')
            torch.save(filterweight.state_dict(), 'pkl/best_filterweight_'+ args.dataname + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print("Early stopping")
            break

    print('Save the model from {}th epoch'.format(best_t + 1))