import argparse
import warnings
import seaborn as sns

import torch
from alive_progress import alive_bar
import random
import numpy as np
import torch as th
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
th.manual_seed(args.seed)
th.cuda.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

from dataset_loader import DataLoader
import time

if __name__ == "__main__":
    print(args)
    # Step 1: Load data =================================================================== #
    num_channels = 3
    feat, edge_index, label, lbl, n_feat, n_classes = load_data_multiple(args.dataname, args.device, num_channels=num_channels)

    # dataname2 = 'texas'
    # feat_2, edge_index_2, label_2, lbl_2, n_feat_2, n_classes_2 = load_data_multiple(dataname2, args.device, num_channels=3)

    # dataname3 = 'cornell'
    # feat_3, edge_index_3, label_3, lbl_3, n_feat_3, n_classes_3 = load_data_multiple(dataname3, args.device, num_channels=3)

    disc = Discriminator(args.hid_dim).to(args.device)

    if 0:
        # Step 2: Create model =================================================================== #
        model = Model(in_dim=n_feat, out_dim=args.hid_dim, K=args.K, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn)
        model = model.to(args.device)

        lbl = lbl.to(args.device)

        # Step 3: Create training components ===================================================== #
        optimizer = torch.optim.Adam([{'params': model.encoder.lin1.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                    {'params': model.disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                                    {'params': model.encoder.prop1.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                                    {'params': model.alpha, 'weight_decay': args.wd, 'lr': args.lr},
                                    {'params': model.beta, 'weight_decay': args.wd, 'lr': args.lr},
                                    #  {'params': model.weight.parameters(), 'weight_decay': args.wd, 'lr': args.lr}
                                    ])
    if 1:
        # Step 2: Create model =================================================================== #
        model = BWGNN(n_feat, args.hid_dim, dprate=args.dprate, dropout=args.dropout, is_bns=args.is_bns, act_fn=args.act_fn, d = num_channels-1)
        model = model.to(args.device)

        lbl = lbl.to(args.device)
        # lbl_2 = lbl_2.to(args.device)
        polyprompt = PolyPrompt(args.hid_dim, num_channels=num_channels).to(args.device)
        polyprompt2 = PolyPrompt(args.hid_dim, num_channels=num_channels).to(args.device)

        # Step 3: Create training components ===================================================== #
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                            {'params': disc.parameters(), 'weight_decay': args.wd1, 'lr': args.lr1},
                            {'params': polyprompt.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
                            {'params': polyprompt2.parameters(), 'weight_decay': args.wd, 'lr': args.lr}
                            ])

        # for name, param in model.named_parameters():
        #     print(name)
        # raise ValueError
    if 0:
        # Step 2: Create model =================================================================== #
        model = GNN(n_feat, args.hid_dim, 2)
        model = model.to(args.device)

        lbl = lbl.to(args.device)

        # Step 3: Create training components ===================================================== #
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd1)

    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float("inf")
    cnt_wait = 0
    best_t = 0

    #generate a random number --> later use as a tag for saved model
    tag = str(int(time.time()))

    # with alive_bar(args.epochs) as bar:
    #     for epoch in range(args.epochs):
    #         model.train()
    #         optimizer.zero_grad()

    #         shuf_idx = np.random.permutation(feat.shape[0])
    #         shuf_feat = feat[shuf_idx, :]

    #         # g = transform_graph(feat, edge_index).to(args.device)
    #         h1 = model(edge_index, feat)
    #         h1_shuffle = model(edge_index, shuf_feat)

    #         h_c = polyprompt(h1)
    #         act = torch.nn.ReLU()
    #         c = act(torch.mean(h_c, dim=0))
    #         out = disc(h1, h1_shuffle, c)

    #         out_final = out
    #         lbl_final = lbl

    #         loss = loss_fn(out_final, lbl_final)

    #         loss.backward()
    #         optimizer.step()

            
    #         if epoch % 20 == 0:
    #             print("Epoch: {0}, Loss: {1:0.4f}".format(epoch, loss.item()))

    #             # weight1 = torch.exp(polyprompt.weight_channels[0].detach().cpu())
    #             # weight2 = torch.exp(polyprompt.weight_channels[1].detach().cpu())
    #             # weight3 = torch.exp(polyprompt.weight_channels[2].detach().cpu())
    #             # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
    #         # print("1:")
    #         # weight1 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[0])
    #         # weight2 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[1])
    #         # weight3 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[2])
    #         # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
    #         # weight1 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[0])
    #         # weight2 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[1])
    #         # weight3 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[2])
    #         # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))

    #         # print("2:")
    #         # weight1 = torch.exp(polyprompt2.weight_channels[0].detach().cpu()[0])
    #         # weight2 = torch.exp(polyprompt2.weight_channels[0].detach().cpu()[1])
    #         # weight3 = torch.exp(polyprompt2.weight_channels[0].detach().cpu()[2])
    #         # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
    #         # weight1 = torch.exp(polyprompt2.weight_channels[1].detach().cpu()[0])
    #         # weight2 = torch.exp(polyprompt2.weight_channels[1].detach().cpu()[1])
    #         # weight3 = torch.exp(polyprompt2.weight_channels[1].detach().cpu()[2])
    #         # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))

    #         if loss < best:
    #             best = loss
    #             best_t = epoch
    #             cnt_wait = 0
    #             th.save(model.state_dict(), 'pkl_tem_向量2/best_model_'+ args.dataname +'.pkl')
    #             th.save(polyprompt.state_dict(), 'pkl_tem_向量2/polyweight_'+ args.dataname + '.pkl')
    #             # th.save(model.state_dict(), 'ablation_pkl/weight_tensor/best_model_'+ args.dataname +'.pkl')
    #             # th.save(polyprompt.state_dict(), 'ablation_pkl/weight_tensor/polyweight_'+ args.dataname + '.pkl')pkl_new
    #             # th.save(polyprompt2.state_dict(), 'pkl_general/polyweight_'+ dataname2 + tag + '.pkl')
    #         else:
    #             cnt_wait += 1

    #         if cnt_wait == args.patience:
    #             print("Early stopping")
    #             break
    #         bar()

    # print('Loading {}th epoch'.format(best_t + 1))

    # model.load_state_dict(th.load('pkl/best_model_'+ args.dataname + tag + '.pkl'))
    # dataname_pretrained = 'cora'
    # model.load_state_dict(th.load('pkl/best_model_'+ args.dataname + '1734018592.pkl'))
    model.load_state_dict(th.load('pkl_tem_向量2/best_model_'+ args.pretrain_dataname + '.pkl'))
    # model.load_state_dict(th.load('ablation_pkl/weight_tensor/best_model_'+ args.pretrain_dataname + '.pkl'))
    # model.load_state_dict(th.load('pkl_triple_gnn/best_model_'+ args.pretrain_dataname + '.pkl'))
    model.eval()
    #embeds = model.get_embedding(edge_index, feat)
    # Step 5:  Linear evaluation ========================================================== #
    print("=== Evaluation ===")
    ''' Linear Evaluation '''
    results = []
    results_f1 = []
    # 10 fixed seeds for random splits from BernNet
    # SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042,
    #          2424918363]
    # train_rate = 0.6
    # val_rate = 0.2
    # percls_trn = int(round(train_rate*len(label)/n_classes))
    # val_lb = int(round(val_rate*len(label)))
    shot_num = 5
    feat_avg = torch.mean(feat,dim=0)
    for i in range(5):
        # time_all = 0.0
        # seed = SEEDS[i]
        # assert label.shape[0] == n_node
        # train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)


        # train_mask = th.BoolTensor(train_mask).to(args.device)
        # val_mask = th.BoolTensor(val_mask).to(args.device)
        # test_mask = th.BoolTensor(test_mask).to(args.device)

        train_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/train_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)
        test_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/test_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)
        val_mask = torch.load("/home/dell/luohaitong/he_prompt/ProG_new/ProG/Experiment/sample_data_new/Node/{}/{}_shot/{}/val_idx.pt".format(args.dataname, shot_num, i)).type(torch.long).to(args.device)

        # train_embs = embeds[train_mask]
        # val_embs = embeds[val_mask]
        # test_embs = embeds[test_mask]

        label = label.to(args.device)

        train_labels = label[train_mask]
        val_labels = label[val_mask]
        test_labels = label[test_mask]

        best_val_f1 = 0
        eval_acc = 0
        eval_f1 = 0
        bad_counter = 0

        logreg = LogReg(hid_dim=args.hid_dim, n_classes=n_classes)
        opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
        logreg = logreg.to(args.device)

        mean = torch.mean(feat, dim=0)
        std = torch.std(feat, dim=0)
        std = torch.clamp(std, min=1e-6)
        # prompt_1 = GPF_plus(n_feat, 10, mean, std)
        # prompt_2 = GPF_plus(n_feat, 10, mean, std)
        # prompt_3 = GPF_plus(n_feat, 10, mean, std)
        # prompt_1 = GPF(n_feat)
        # prompt_2 = GPF(n_feat)
        # prompt_3 = GPF(n_feat)
        #prompt = GPF(n_feat)
        #cora init_cross_prune 0.5 cross_prune 0.6
        if args.dataname in ['cora','pubmed','citeseer']:
            cross_prune = 0.55  #0.6
        elif args.dataname in ['wisconsin']:
            cross_prune = 0.4  # 0.4
        elif args.dataname in ['amazon_ratings']:
            cross_prune = 0.4
        else:
            cross_prune = 0.5  # 0.4
        cross_prune = args.threshold
        prompt_1 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2)
        prompt_2 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2)
        prompt_3 = HeavyPrompt(n_feat, args.prompt_num, mean, std, cross_prune=cross_prune, inner_prune=0.2)

        polyprompt = PolyPrompt(args.hid_dim, num_channels=num_channels)
        # original_state_dict = model.state_dict()
        # new_state_dict = {}
        # for key, value in original_state_dict.items():
        #     if ('weight_channels' in key):
        #         new_state_dict[key] = value
        # polyprompt.load_state_dict(new_state_dict)
        # polyprompt.load_state_dict(th.load('pkl/polyweight_'+ args.dataname + '1734018592.pkl'))
        polyprompt.load_state_dict(th.load('pkl_tem_向量2/polyweight_'+ args.pretrain_dataname + '.pkl'))
        # polyprompt.load_state_dict(th.load('ablation_pkl/weight_tensor/polyweight_'+ args.pretrain_dataname + '.pkl'))
        # polyprompt.load_state_dict(th.load('pkl_triple_gnn/polyweight_'+ args.pretrain_dataname + '.pkl'))
        polyprompt = polyprompt.to(args.device)

        # opt_prompt = th.optim.Adam(prompt.parameters(), lr=args.lr2, weight_decay=args.wd2)
        # opt_prompt = th.optim.Adam(list(prompt_1.parameters()) + list(prompt_2.parameters()) + list(prompt_3.parameters()) + list(polyprompt.parameters()), lr=args.lr2, weight_decay=args.wd2)
        # opt_prompt = th.optim.Adam(list(prompt_1.parameters()) + list(prompt_2.parameters()) + list(prompt_3.parameters()), lr=args.lr2, weight_decay=args.wd2)
        # opt_prompt = th.optim.Adam(polyprompt.parameters(), lr=args.lr1, weight_decay=args.wd1)
        if args.mode == '1' or args.mode == '4' or args.mode == '5':
            opt_prompt = torch.optim.Adam([{'params': prompt_1.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
                        {'params': prompt_2.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
                        {'params': prompt_3.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2}
                        ])
        else:
            opt_prompt = torch.optim.Adam([{'params': prompt_1.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
                                    {'params': prompt_2.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2},
                                    {'params': prompt_3.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2}
                                    ])
            opt_polyprompt = torch.optim.Adam([{'params': polyprompt.parameters(), 'weight_decay': args.wd2, 'lr': args.lr2}
                                ])
        
        prompt_1 = prompt_1.to(args.device)
        prompt_2 = prompt_2.to(args.device)
        prompt_3 = prompt_3.to(args.device)

        loss_fn = nn.CrossEntropyLoss()

        # g = transform_graph(feat, edge_index).to(args.device)
        # edge_index = torch.unique(torch.cat([edge_index, edge_index.flip(0)],dim=1), dim=1)
        flag = True
        for epoch in tqdm(range(2000)):
            #time_s = time.time()
            logreg.train()
            prompt_1.train()
            prompt_2.train()
            prompt_3.train()
            if args.mode == '1' or args.mode == '4' or args.mode == '5':
                polyprompt.eval()
            else:
                polyprompt.train()
                opt_polyprompt.zero_grad()
            opt.zero_grad()
            opt_prompt.zero_grad()
            # if epoch == 0:
            #     edge_index_1, feat_prompted1 = prompt_1.add(feat, edge_index, 0.5)
            #     edge_index_2, feat_prompted2 = prompt_2.add(feat, edge_index, 0.5)
            #     edge_index_3, feat_prompted3 = prompt_3.add(feat, edge_index, 0.5)
            # else:
            #     edge_index_1, feat_prompted1 = prompt_1.add(feat, edge_index)
            #     edge_index_2, feat_prompted2 = prompt_2.add(feat, edge_index)
            #     edge_index_3, feat_prompted3 = prompt_3.add(feat, edge_index)
            edge_index_1, feat_prompted1 = prompt_1.add(feat, edge_index)
            edge_index_2, feat_prompted2 = prompt_2.add(feat, edge_index)
            edge_index_3, feat_prompted3 = prompt_3.add(feat, edge_index)

            if not(args.mode == '1' or args.mode == '4' or args.mode == '5'):
                if flag:
                    prompt_1.train()
                    prompt_2.train()
                    prompt_3.train()
                    polyprompt.eval()
                else:
                    prompt_1.eval()
                    prompt_2.eval()
                    prompt_3.eval()
                    polyprompt.train()

            if args.mode == '2' or args.mode == '4':
                embeds = model.get_embedding(edge_index, edge_index, edge_index, feat, feat, feat)
                embeds = polyprompt(embeds)
            elif args.mode == '3' or args.mode == '5':
                embeds = model.get_embedding(edge_index_1, edge_index_1, edge_index_1, feat_prompted1, feat_prompted1, feat_prompted1)
                embeds = polyprompt(embeds)[args.prompt_num:]
            elif args.mode == '0' or args.mode == '1':
                embeds = model.get_embedding(edge_index_1, edge_index_2, edge_index_3, feat_prompted1, feat_prompted2, feat_prompted3)
                embeds = polyprompt(embeds)[args.prompt_num:]

            train_embs = embeds[train_mask]
            val_embs = embeds[val_mask]
            test_embs = embeds[test_mask]

            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            if not(args.mode == '1' or args.mode == '4' or args.mode == '5'):
                if flag:
                    opt_prompt.step()
                    flag=False
                else:
                    opt_polyprompt.step()
                    flag=True
            else:
                opt_prompt.step()

            # print("Epoch: {0}, Loss: {1:0.9f}".format(epoch, loss.item()))
            logreg.eval()
            # prompt_1.eval()
            # prompt_2.eval()
            # prompt_3.eval()
            # polyprompt.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                val_f1 = f1_score(val_labels.cpu(), val_preds.cpu(), average='macro')
                test_f1 = f1_score(test_labels.cpu(), test_preds.cpu(), average='macro')

                if val_f1 >= best_val_f1:
                    bad_counter = 0
                    best_val_f1 = val_f1
                    # if test_acc > eval_acc:
                    #     eval_acc = test_acc
                    #     eval_f1 = test_f1
                    eval_acc = test_acc
                    eval_f1 = test_f1
                    torch.save(feat_prompted1, f'./prompted_graph/_{args.dataname}_feat_prompt1.pt')
                    torch.save(feat_prompted2, f'./prompted_graph/_{args.dataname}_feat_prompt2.pt')
                    torch.save(feat_prompted3, f'./prompted_graph/_{args.dataname}_feat_prompt3.pt')
                    torch.save(edge_index_1, f'./prompted_graph/_{args.dataname}_edge_index1.pt')
                    torch.save(edge_index_2, f'./prompted_graph/_{args.dataname}_edge_index2.pt')
                    torch.save(edge_index_3, f'./prompted_graph/_{args.dataname}_edge_index3.pt')
                else:
                    bad_counter += 1
            # time_e = time.time()
            # time_all += time_e-time_s

        # torch.save(feat_prompted1, 'feat_prompt1_2.pt')
        # torch.save(feat_prompted2, 'feat_prompt2_2.pt')
        # torch.save(feat_prompted3, 'feat_prompt3_2.pt')
        print(i, 'Linear evaluation accuracy:{:.4f} | f1:{:.4f}'.format(eval_acc, eval_f1))
        #print('time:{:.4f}'.format(time_all/2000*1000))
        # weight1 = torch.exp(polyprompt.weight_channels[0].detach().cpu())
        # weight2 = torch.exp(polyprompt.weight_channels[1].detach().cpu())
        # weight3 = torch.exp(polyprompt.weight_channels[2].detach().cpu())
        # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
        # weight1 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[0])
        # weight2 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[1])
        # weight3 = torch.exp(polyprompt.weight_channels[0].detach().cpu()[2])
        # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
        # weight1 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[0])
        # weight2 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[1])
        # weight3 = torch.exp(polyprompt.weight_channels[1].detach().cpu()[2])
        # print(weight1/(weight1+weight2+weight3),weight2/(weight1+weight2+weight3),weight3/(weight1+weight2+weight3))
        # print(i, 'Linear evaluation f1:{:.4f}'.format(eval_f1))
        results.append(eval_acc.cpu().data)
        results_f1.append(eval_f1)

    results = [v.item() for v in results]
    test_acc_mean = np.mean(results, axis=0)
    test_acc_std = np.std(results, axis=0)
    print(f'test acc mean = {test_acc_mean:.4f} ± {test_acc_std:.4f}')
    # values = np.asarray(results, dtype=object)
    # acc_uncertainty = np.max(
    #     np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    # print(f'test acc mean = {test_acc_mean:.4f} ± {acc_uncertainty:.4f}')

    results_f1 = [v.item() for v in results_f1]
    test_f1_mean = np.mean(results_f1, axis=0)
    test_f1_std = np.std(results_f1, axis=0)
    print(f'test f1 mean = {test_f1_mean:.4f} ± {test_f1_std:.4f}')
    # values = np.asarray(results_f1, dtype=object)
    # f1_uncertainty = np.max(
    #     np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
    # print(f'test f1 mean = {test_f1_mean:.4f} ± {f1_uncertainty:.4f}')


    result_dict = {
    'mode': args.mode,
    'pretrain_dataset': args.pretrain_dataname,
    'downstream_dataset': args.dataname,
    'thresold': args.threshold,
    'Acc_mean': round(test_acc_mean, 4),
    'Acc_std': round(test_acc_std, 4),
    'F1_mean': round(test_f1_mean, 4),
    'F1_std': round(test_f1_std, 4)
    }


    # with open('./results/向量7_correct.csv', mode='a', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=result_dict.keys())
    #     if file.tell() == 0:
    #         writer.writeheader()
    #     writer.writerow(result_dict)
    with open('./results/阈值实验.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(result_dict)

