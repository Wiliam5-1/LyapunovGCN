"""
Dataset loading utilities for standard graph benchmarks
"""
import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import linalg
import sys
import os


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_planetoid_data(dataset_str):
    """
    Load citation network dataset (Cora, Citeseer, Pubmed)
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _remap_indices(idx, mask, mapping):
    """Filter indices by mask and remap to new contiguous ids."""
    if idx.numel() == 0:
        return idx
    keep = mask[idx]
    if keep.sum() == 0:
        return idx.new_empty((0,), dtype=idx.dtype)
    return mapping[idx[keep]]


def _make_binary_subgraph(adj, features, labels, idx_train, idx_val, idx_test, classes=(0, 1)):
    """
    Filter graph to two classes and remap labels to {0,1}.
    """
    class0, class1 = classes
    mask = (labels == class0) | (labels == class1)
    keep_nodes = mask.nonzero(as_tuple=False).squeeze()
    mapping = torch.full((labels.size(0),), -1, device=labels.device, dtype=torch.long)
    mapping[keep_nodes] = torch.arange(keep_nodes.numel(), device=labels.device)

    # Filter features and labels
    features = features[mask]
    labels = labels[mask].clone()
    labels[labels == class0] = 0
    labels[labels == class1] = 1

    # Filter adjacency
    if adj.is_sparse:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        src = indices[0]
        dst = indices[1]
        keep_edges = mask[src] & mask[dst]
        new_src = mapping[src[keep_edges]]
        new_dst = mapping[dst[keep_edges]]
        new_indices = torch.stack([new_src, new_dst], dim=0)
        new_values = values[keep_edges]
        new_size = (keep_nodes.numel(), keep_nodes.numel())
        adj = torch.sparse_coo_tensor(new_indices, new_values, new_size, device=adj.device)
    else:
        adj = adj[mask][:, mask]

    # Remap train/val/test indices
    idx_train = _remap_indices(idx_train, mask, mapping)
    idx_val = _remap_indices(idx_val, mask, mapping)
    idx_test = _remap_indices(idx_test, mask, mapping)

    return adj, features, labels, idx_train, idx_val, idx_test


def resplit_dataset(num_nodes, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    重新划分数据集为train/val/test
    
    Args:
        num_nodes: 节点总数
        labels: 标签
        train_ratio: 训练集比例 (default: 0.7)
        val_ratio: 验证集比例 (default: 0.2)
        test_ratio: 测试集比例 (default: 0.1)
        seed: 随机种子
    
    Returns:
        idx_train, idx_val, idx_test
    """
    import numpy as np
    np.random.seed(seed)
    
    # 获取所有节点索引
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    # 计算分割点
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    # 分割
    idx_train = torch.LongTensor(indices[:train_size])
    idx_val = torch.LongTensor(indices[train_size:train_size + val_size])
    idx_test = torch.LongTensor(indices[train_size + val_size:])
    
    return idx_train, idx_val, idx_test


def load_data(dataset_name, use_standard_split=False, seed=42):
    """
    Load dataset and return torch tensors
    
    Args:
        dataset_name: one of ['cora', 'citeseer', 'pubmed', 'corafull', 'cs', 'physics', 'photo', 'computers',
                              'chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin']
        use_standard_split: If False, use K-fold CV; If True, use standard split (default: False)
    
    Returns:
        adj, features, labels, idx_train, idx_val, idx_test
    """
    # Support binary dataset variants (e.g., cora_binary)
    dataset_name = dataset_name.lower()
    if dataset_name.endswith('_binary'):
        base_name = dataset_name.replace('_binary', '')
        binary_mode = True
    else:
        base_name = dataset_name
        binary_mode = False

    # Dataset type mapping
    # Planetoid: cora, citeseer, pubmed
    # Coauthor: cs, physics
    # Amazon: photo, computers
    # CitationFull: corafull
    
    # Try to load from pytorch geometric
    try:
        from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull, WikipediaNetwork, WebKB
        import torch_geometric.transforms as T
        
        # Load dataset based on type
        if base_name in ['cora', 'citeseer', 'pubmed']:
            # Planetoid datasets (standard citation networks)
            dataset = Planetoid(root=f'data/{base_name}', name=base_name.capitalize())
        elif base_name == 'corafull':
            # Full Cora dataset (larger)
            dataset = CitationFull(root='data/corafull', name='Cora')
        elif base_name in ['cs', 'physics']:
            # Coauthor datasets
            dataset = Coauthor(root=f'data/{base_name}', name=base_name.upper())
        elif base_name in ['photo', 'computers']:
            # Amazon datasets
            dataset = Amazon(root=f'data/{base_name}', name=base_name.capitalize())
        elif base_name in ['chameleon', 'squirrel']:
            # WikipediaNetwork datasets
            dataset = WikipediaNetwork(root=f'data/{base_name}', name=base_name)
        elif base_name in ['cornell', 'texas', 'wisconsin']:
            # WebKB datasets
            dataset = WebKB(root=f'data/{base_name}', name=base_name.capitalize())
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        data = dataset[0]
        
        # Convert to required format
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Create adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1
        
        # Make symmetric
        adj = adj + adj.t()
        adj[adj > 1] = 1
        
        # Convert to sparse
        adj = adj.to_sparse()
        
        features = data.x
        labels = data.y
        
        # Get train/val/test masks
        # Check if masks exist, otherwise generate them
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            idx_train = data.train_mask.nonzero(as_tuple=False).squeeze()
            idx_val = data.val_mask.nonzero(as_tuple=False).squeeze()
            idx_test = data.test_mask.nonzero(as_tuple=False).squeeze()
        else:
            # Generate splits for datasets without predefined masks
            print(f"Dataset {base_name} has no predefined splits, generating new splits...")
            num_nodes = features.shape[0]
            idx_train, idx_val, idx_test = resplit_dataset(num_nodes, labels, seed=seed)
        
        if binary_mode:
            adj, features, labels, idx_train, idx_val, idx_test = _make_binary_subgraph(
                adj, features, labels, idx_train, idx_val, idx_test, classes=(0, 1)
            )
        
        # 重新划分数据集（7:2:1）
        if not use_standard_split:
            num_nodes = features.shape[0]
            idx_train, idx_val, idx_test = resplit_dataset(num_nodes, labels, seed=seed)
        
        return adj, features, labels, idx_train, idx_val, idx_test
        
    except Exception as e:
        print(f"PyTorch Geometric loading failed: {e}")
        print("Falling back to manual loading...")
        
        # Fallback to manual loading
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
            load_planetoid_data(base_name)
        
        # Preprocess features
        features = preprocess_features(features)
        features = torch.FloatTensor(features)
        
        # Convert adjacency matrix to torch sparse tensor
        adj = sp.coo_matrix(adj)
        indices = torch.from_numpy(
            np.vstack((adj.row, adj.col)).astype(np.int64))
        values = torch.from_numpy(adj.data.astype(np.float32))
        shape = torch.Size(adj.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
        
        # Convert labels
        labels = torch.from_numpy(np.where(y_train + y_val + y_test)[1])
        
        # Get indices
        idx_train = torch.from_numpy(np.where(train_mask)[0])
        idx_val = torch.from_numpy(np.where(val_mask)[0])
        idx_test = torch.from_numpy(np.where(test_mask)[0])
        
        if binary_mode:
            adj, features, labels, idx_train, idx_val, idx_test = _make_binary_subgraph(
                adj, features, labels, idx_train, idx_val, idx_test, classes=(0, 1)
            )
        
        # 重新划分数据集（7:2:1）
        if not use_standard_split:
            num_nodes = features.shape[0]
            idx_train, idx_val, idx_test = resplit_dataset(num_nodes, labels, seed=seed)
        
        return adj, features, labels, idx_train, idx_val, idx_test


def get_dataset_config(dataset_name):
    """
    Get optimal hyperparameters for each dataset based on literature
    """
    configs = {
        'cora': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'citeseer': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'pubmed': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'chameleon': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'squirrel': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'wisconsin': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'physics': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'texas': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'cornell': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'cora_binary': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'citeseer_binary': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        },
        'pubmed_binary': {
            'lr': 0.01,
            'weight_decay': 5e-4,
            'hidden': 64,
            'dropout': 0.5,
            'epochs': 200,
            'patience': 100
        }
    }
    
    return configs.get(dataset_name.lower(), configs['cora'])
