import argparse
import warnings
import seaborn as sns

import torch
from alive_progress import alive_bar
import random
import numpy as np
import torch.nn as nn
from utils import random_splits, transform_graph
import os
warnings.filterwarnings("ignore")

from model import LogReg,Model
from GPF import GPF,GPF_plus
from prompt import PolyPrompt, PolyPrompt_tensor, PolyPrompt_concat
# from BWGNN import BWGNN
from AllInOnePrompt import HeavyPrompt
from BWGNN_multiple import BWGNN, Discriminator, Triple_GNN
from tqdm import *
from GCN import GNN
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from utils import load_data_multiple
from sklearn.preprocessing import StandardScaler
import csv
import os.path as osp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description="PolyGCL")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # Default seed same as GCNII
parser.add_argument('--dev', type=int, default=0, help='device id')

parser.add_argument(
    "--dataname", type=str, default="cora", help="Name of dataset."
)
parser.add_argument(
    "--mode", type=str, default="1", help="0:full, 1:w/o weight tuning, 2:w/o prompt, 3:only single prompt, 4:w/o weight tuning w/o prompt, 5:w/o weight tuning only single prompt."
)
parser.add_argument(
    "--pretrain_dataname", type=str, default="cora", help="Name of dataset."
)
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu."
)
parser.add_argument("--epochs", type=int, default=500, help="Training epochs.")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="Patient epochs to wait before early stopping.",
)
parser.add_argument(
    "--lr", type=float, default=0.010, help="Learning rate of prop."
)
parser.add_argument(
    "--lr1", type=float, default=0.001, help="Learning rate of PolyGCL."
)
parser.add_argument(
    "--lr2", type=float, default=0.005, help="Learning rate of linear evaluator."
)
parser.add_argument(
    "--wd", type=float, default=0.0, help="Weight decay of PolyGCL prop."
)
parser.add_argument(
    "--wd1", type=float, default=0.0, help="Weight decay of PolyGCL."
)
parser.add_argument(
    "--wd2", type=float, default=0.0, help="Weight decay of linear evaluator."
)

parser.add_argument(
    "--hid_dim", type=int, default=128, help="Hidden layer dim."
)

parser.add_argument(
    "--K", type=int, default=10, help="Layer of encoder."
)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu',
                    help='activation function')
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Layer of encoder."
)
parser.add_argument(
    "--prompt_num", type=int, default=20, help="Layer of encoder."
)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

from dataset_loader import DataLoader
import time

if __name__ == "__main__":
    print(args)
    # Step 1: Load data =================================================================== #
    num_channels = 3
    feat, edge_index, label, lbl, n_feat, n_classes = load_data_multiple(args.dataname, args.device, num_channels=num_channels)

    disc = Discriminator(args.hid_dim).to(args.device)

    # Step 2: Create model =================================================================== #
    model = BWGNN(n_feat, args.hid_dim, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn, d = num_channels-1)
    model = model.to(args.device)

    lbl = lbl.to(args.device)
    polyprompt = PolyPrompt(args.hid_dim, num_channels=num_channels).to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                        {'params': disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                        {'params': polyprompt.parameters(), 'weight_decay': args.wd, 'lr': args.lr}
                        ])

    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    best_t = 0

    #generate a random number --> later use as a tag for saved model
    tag = str(int(time.time()))

    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            shuf_idx = np.random.permutation(feat.shape[0])
            shuf_feat = feat[shuf_idx, :]

            h1 = model(edge_index, feat)
            h1_shuffle = model(edge_index, shuf_feat)

            h_c = polyprompt(h1)
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
                torch.save(polyprompt.state_dict(), 'pkl/polyweight_'+ args.dataname + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()

    print('Save the model from {}th epoch'.format(best_t + 1))

