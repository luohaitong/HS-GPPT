import argparse
import warnings

import torch
import random
import numpy as np
import torch.nn as nn
import os

from model import LogReg
from prompt import PolyPrompt

from PromptGraph import HeavyPrompt
from BWGNN_multiple import BWGNN, Triple_GNN
from tqdm import *
from sklearn.metrics import f1_score
from utils import load_data_multiple

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description="HS-GPPT")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--mode", type=str, default="0", help="0:full, 1: singe prompt, 2: w/o prompt, 3: w/o prompt norm")
parser.add_argument("--pretrain_dataname", type=str, default="cora", help="Name of dataset.")
parser.add_argument("--dataname", type=str, default="cora", help="Name of dataset.")

parser.add_argument("--gpu", type=int, default=0, help="GPU index. Default: -1, using cpu.")

parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
parser.add_argument("--lr", type=float, default=0.005, help="Learning rate of linear evaluator.")
parser.add_argument("--wd", type=float, default=0.0, help="Weight decay of linear evaluator.")
parser.add_argument("--hid_dim", type=int, default=128, help="Hidden layer dim.")

parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

parser.add_argument('--is_bns', type=bool, default=False)
parser.add_argument('--act_fn', default='relu',help='activation function')
parser.add_argument("--threshold", type=float, default=0.5, help="Layer of encoder.")
parser.add_argument("--prompt_num", type=int, default=10, help="Layer of encoder.")

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
    num_fitlers = 3
    shot_num = 5

    feat, edge_index, label, lbl, n_feat, n_classes = load_data_multiple(args.dataname, args.device, num_channels=num_fitlers)

    # Step 2: Load model =================================================================== #
    model = BWGNN(n_feat, args.hid_dim, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn, d = num_fitlers-1).to(args.device)
    polyprompt = PolyPrompt(args.hid_dim, num_channels=num_fitlers).to(args.device)

    model.load_state_dict(torch.load('pkl/best_model_'+ args.pretrain_dataname + '.pkl'))
    polyprompt.load_state_dict(torch.load('pkl/polyweight_'+ args.pretrain_dataname + '.pkl'))

    model.eval()
    polyprompt.eval()

    # Step 5:  Evaluation ========================================================== #
    results = []
    results_f1 = []

    feat_avg = torch.mean(feat,dim=0)
    mean = torch.mean(feat, dim=0)
    std = torch.clamp(torch.std(feat, dim=0), min=1e-6)

    for i in range(5):

        train_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/train_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)
        test_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/test_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)
        val_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/val_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_f1 = 0.0
        eval_acc = 0.0
        eval_f1 = 0.0

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = torch.optim.Adam(logreg.parameters(), lr=args.lr, weight_decay=args.wd)
        logreg = logreg.to(args.device)

        # Construct prompt graphs =================================================================== #
        if args.dataname in ['cora','pubmed','citeseer']:
            cross_prune = 0.55  #0.6
        elif args.dataname in ['wisconsin']:
            cross_prune = 0.4  # 0.4
        elif args.dataname in ['amazon_ratings']:
            cross_prune = 0.4
        else:
            cross_prune = 0.5
        cross_prune = args.threshold
        prompt_1 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2).to(args.device)
        prompt_2 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2).to(args.device)
        prompt_3 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2).to(args.device)

        opt_prompt = torch.optim.Adam([{'params': prompt_1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                {'params': prompt_2.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                {'params': prompt_3.parameters(), 'weight_decay': args.wd, 'lr': args.lr}
                                ])

        logreg.train()
        prompt_1.train()
        prompt_2.train()
        prompt_3.train()

        loss_fn = nn.CrossEntropyLoss()

        for epoch in tqdm(range(args.epochs)):

            logreg.train()
            opt.zero_grad()
            opt_prompt.zero_grad()

            edge_index_1, feat_prompted1 = prompt_1.add(feat, edge_index)
            edge_index_2, feat_prompted2 = prompt_2.add(feat, edge_index)
            edge_index_3, feat_prompted3 = prompt_3.add(feat, edge_index)

            if args.mode == '0' or args.mode == '3':
                embeds = model.get_embedding(edge_index_1, edge_index_2, edge_index_3, feat_prompted1, feat_prompted2, feat_prompted3)
                embeds = polyprompt(embeds)[args.prompt_num:]
            elif args.mode == '1':
                embeds = model.get_embedding(edge_index_1, edge_index_1, edge_index_1, feat_prompted1, feat_prompted1, feat_prompted1)
                embeds = polyprompt(embeds)[args.prompt_num:]
            elif args.mode == '2':
                embeds = model.get_embedding(edge_index, edge_index, edge_index, feat, feat, feat)
                embeds = polyprompt(embeds)

            train_embs = embeds[train_mask]
            val_embs = embeds[val_mask]
            test_embs = embeds[test_mask]

            logits = logreg(train_embs)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()
            opt_prompt.step()

            logreg.eval()
            with torch.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                val_f1 = f1_score(val_labels.cpu(), val_preds.cpu(), average='macro')
                test_f1 = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro')

                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1
                    eval_acc = test_acc
                    eval_f1 = test_f1


        print(i, 'Evaluation accuracy:{:.4f} | f1:{:.4f}'.format(eval_acc, eval_f1))

        results.append(eval_acc.cpu().data)
        results_f1.append(eval_f1)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0)
    test_acc_std = np.std(results, axis=0)
    print(f'test acc mean = {test_acc_mean:.4f} ± {test_acc_std:.4f}')


    results_f1 = [v.item() for v in results_f1]
    test_f1_mean = np.mean(results_f1, axis=0)
    test_f1_std = np.std(results_f1, axis=0)
    print(f'test f1 mean = {test_f1_mean:.4f} ± {test_f1_std:.4f}')