import numpy as np
import torch

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25):
    """ randomly splits label into train/valid/test splits """
    if len(label.shape) > 1 and label.shape[1] > 1:
        labeled_nodes = torch.arange(label.shape[0])
    else:
        labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    np.random.seed(42)
    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    print(train_idx.shape, valid_idx.shape, test_idx.shape)

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, tr_num_per_class=20, val_num_per_class=30):
    label = label.cpu()
    train_idx, valid_idx, test_idx = [], [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:tr_num_per_class].tolist()
        valid_idx += rand_idx[tr_num_per_class:tr_num_per_class+val_num_per_class].tolist()
        test_idx += rand_idx[tr_num_per_class+val_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    valid_idx = torch.as_tensor(valid_idx)
    test_idx = torch.as_tensor(test_idx)
    test_idx = test_idx[torch.randperm(test_idx.shape[0])]

    return train_idx, valid_idx, test_idx



class NCDataset(object):
    def __init__(self, name):
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, tr_num_per_class=20,
                      val_num_per_class=30):
        if split_type == 'random_all':
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'random_class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, tr_num_per_class=tr_num_per_class,
                                                               val_num_per_class=val_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(dataname):
    if dataname in ("acm", "dblp"):
        dataset = load_citation_dataset(dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_citation_dataset(name):
    root = "./dataset/" + name
    data = torch.load(root + "/data.pt")[0]
    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset
