import dgl
import torch
import numpy as np
from dgl.data import CoraFullDataset,CoauthorCSDataset,CoauthorPhysicsDataset

def get_train_val_test_split2(random_state,num_samples : int,
                             train_rate : float, val_rate : float):
    assert 0 < train_rate < 1
    assert 0 < val_rate < 1
    assert 0 < train_rate+val_rate < 1
    
    train_size = int(num_samples * train_rate)
    val_size = int(num_samples * val_rate)
    
    remaining_indices = list(range(num_samples))
    # select train examples with no respect to class distribution
    train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    remaining_indices = np.setdiff1d(remaining_indices, train_indices)
    val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    print(len(set(val_indices)), len(val_indices))
    print(len(set(test_indices)), len(test_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))

    return train_indices, val_indices, test_indices

def gen_dataset(dataset):
    if dataset == "corafull":
        data = CoraFullDataset()
    elif dataset == "cs":
        data = CoauthorCSDataset()
    elif dataset == "physics":
        data = CoauthorPhysicsDataset()
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    g = data[0]
    feat = g.ndata['feat']  # get node feature
    label = g.ndata['label']  # get node labels
    datalen = g.num_nodes()  # get number of nodes
    print(f'{dataset} has {datalen} nodes')
    random_state = np.random.RandomState(0)
    idx_train, idx_val, idx_test = get_train_val_test_split2(random_state, datalen, train_rate=0.6, val_rate=0.2)
    
    data_list = [g.adj(), feat, label, idx_train, idx_val, idx_test]
    torch.save(data_list,f'./dataset/{dataset}.pt')
    print('saved training data')

if __name__ == "__main__":
    gen_dataset("corafull")
    gen_dataset("cs")
    gen_dataset("physics")