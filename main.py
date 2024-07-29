import argparse
import numpy as np
from torch_geometric.utils import to_undirected, subgraph, k_hop_subgraph
from logger import Logger
from dataset import load_dataset
from data_utils import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parser_add_main_args
import warnings
from model import *
import pymetis
warnings.filterwarnings('ignore')
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(42)

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

dataset_names = args.dataset.split(",")
print(dataset_names)
datasets = []

for dataset_name in dataset_names:
    datasets.append(load_dataset(dataset_name))

for dataset in datasets:
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

split_idx_lsts = []
partition_list = []
for i in range(len(datasets)):
    dataset = datasets[i]
    adj = []
    for k in range(dataset.graph["node_feat"].shape[0]):
        adj.append(dataset.graph["edge_index"][1][dataset.graph["edge_index"][0] == k].tolist())
    (edgecuts, parts) = pymetis.part_graph(nparts=args.nparts, adjacency=adj)
    g_parts = []
    for w in range(args.nparts):
        g_parts.append(torch.argwhere(torch.tensor(parts) == w).flatten())
    partition_list.append(g_parts)
    dataset_name = dataset_names[i]
    if args.rand_split:
        if i < len(datasets) - args.downstream_num:
            split_idx_lst = dataset.get_idx_split(split_type='random_all', train_prop=1.,
                                                  valid_prop=0.)
        else:
            split_idx_lst = dataset.get_idx_split(split_type='random_all', train_prop=0.0,
                                                  valid_prop=0.)
    elif args.rand_split_class:
        if i < len(datasets) - args.downstream_num:
            split_idx_lst = dataset.get_idx_split(split_type='random_class', tr_num_per_class=args.tr_num_per_class, val_num_per_class=args.val_num_per_class)
        else:
            split_idx_lst = dataset.get_idx_split(split_type='random_class', tr_num_per_class=1,
                                                  val_num_per_class=args.val_num_per_class)
    split_idx_lsts.append(split_idx_lst)


model = None
nodes_num = []
class_num = []
f_dim = []
for i in range(len(datasets)):
    dataset = datasets[i]
    dataset_name = dataset_names[i]
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    e = dataset.graph['edge_index'].shape[1]
    nodes_num.append(n)
    class_num.append(c)
    f_dim.append(d)
    if i == 0:
        model = SPA(d, args.hidden_channels, c, args, n).to(device)
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    print(dataset_name + ":")
    print(f"num nodes {n} | num classes {c} | num node feats {d} | num edges {e}")


criterion = nn.NLLLoss()
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

loggers = []
for i in range(len(datasets)):
    loggers.append(Logger(args.runs, args))

for i in range(len(datasets)):
    dataset = datasets[i]
    dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=nodes_num[i])


weights = []
for i in range(args.nparts):
    weights.append(1/args.nparts)
for run in range(args.runs):
    split_indice = []
    train_indice = []
    valid_indice = []
    for i in range(len(datasets)):
        split_idx_lst = split_idx_lsts[i]
        dataset = datasets[i]
        dataset_name = dataset_names[i]
        split_idx = split_idx_lst
        split_indice.append(split_idx)
        train_idx = split_idx['train'].to(device)
        train_indice.append(train_idx)
        valid_idx = torch.cat([train_idx, split_idx['valid'].to(device)], dim=-1)
        valid_indice.append(valid_idx)
        dataset.graph['tr_edge_index'] = subgraph(train_idx.cpu(), dataset.graph['edge_index'].cpu())[0]   # edge index for tr_node
        dataset.graph['va_edge_index'] = subgraph(valid_idx, dataset.graph['edge_index'])[0]   # edge index for tr_node and va_node


    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    for epoch in range(args.epochs):
        for i in range(len(datasets)):
            g_parts = partition_list[i]
            idx = torch.argsort(torch.concat(g_parts))
            model.train()
            optimizer.zero_grad()
            dataset = datasets[i]
            dataset_name = dataset_names[i]
            print("Training for the dataset : " + dataset_name)
            split_idx = split_indice[i]
            train_idx = train_indice[i]
            valid_idx = valid_indice[i]
            logger = loggers[i]
            final_feat = []
            for m in range(dataset.graph["node_feat"].shape[0]):
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(m, args.ego_hop, dataset.graph['edge_index'], relabel_nodes=True)
                out = model(dataset.graph['node_feat'][subset], edge_index)[mapping]
                final_feat.append(out)
            out = torch.concat(final_feat, axis=0)
            if i == 0:
                out = F.log_softmax(out, dim=1)
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1).to(torch.float).squeeze()
                loss = nn.KLDivLoss(reduction='none')(out[train_idx], true_label[train_idx])
                loss = loss.sum(axis=1)
                total_loss = 0
                new_weights = []
                for l in range(len(weights)):
                    total_loss += weights[l] * loss[partition_list[0][l]].mean()
                    new_weights.append(loss[partition_list[0][l]].mean().cpu().detach())
                loss = total_loss
                loss.backward()
                optimizer.step()
                new_weights = [(i - max(new_weights))/10 for i in new_weights]
                weights = torch.softmax(torch.tensor(new_weights, dtype=torch.float), dim=0).tolist()
            else:
                out = F.log_softmax(out, dim=1)
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1).to(torch.float).squeeze()
                loss = nn.KLDivLoss(reduction='batchmean')(out[train_idx], true_label[train_idx])
                loss.backward()
                optimizer.step()

            result = evaluate(dataset_name, model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')
                if args.print_prop:
                    pred = out.argmax(dim=-1, keepdim=True)
                    print("Predicted proportions:", pred.unique(return_counts=True)[1].float() / pred.shape[0])

    for i in range(len(datasets)):
        loggers[i].print_statistics(run)

for i in range(len(datasets)):
    results = loggers[i].print_statistics()
