# Data Downloader and to_scipy_sparse_matrix
import torch_geometric
from torch_geometric.utils import to_scipy_sparse_matrix

# PyTorch
import torch
import torch.nn as nn

# Tool
import scipy.sparse as sparse
import numpy as np


class DataSet:
    features = 0
    labels = 0
    train_labels = 0
    val_labels = 0
    test_labels = 0
    adjacency_matrix = 0
    laplacian_matrix = 0
    num_classes = 0


def data_load(data_path):
    # Download Data
    cora_dataset = torch_geometric.datasets.Planetoid(root=data_path, name="Cora")
    data = cora_dataset[0]
    output = DataSet

    # Features
    features = data.x.clone()
    features_sum = features.sum(1).unsqueeze(1)
    features_sum[features_sum == 0] = 1.0
    features = torch.div(features, features_sum)

    # Read train, test, valid labels
    ignore_index = nn.CrossEntropyLoss().ignore_index  # = -100, used to ignore not allowed labels in CE loss
    num_classes = len(set(data.y.numpy()))
    labels = data.y.clone()
    train_labels = set_labels(data.y.clone(), data.train_mask, ignore_index)
    val_labels = set_labels(data.y.clone(), data.val_mask, ignore_index)
    test_labels = set_labels(data.y.clone(), data.test_mask, ignore_index)

    # Read & normalize adjacency matrix
    adjacency_matrix, adj_csr = ajacency_matrix(data.edge_index)

    # Output
    output.features         = features
    output.num_classes      = num_classes
    output.labels           = labels
    output.train_labels     = train_labels
    output.val_labels       = val_labels
    output.test_labels      = test_labels
    output.adjacency_matrix = adjacency_matrix
    output.laplacian_matrix = get_laplacian_matrix(adj_csr)

    return output


def set_labels(_initial_labels, _set_mask, _ignore_label):
    _initial_labels[~_set_mask] = _ignore_label
    return _initial_labels


def ajacency_matrix(_edge_index):
    adj = to_scipy_sparse_matrix(_edge_index)
    adj += sparse.eye(adj.shape[0])
    degree_for_norm = sparse.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())  # D^(-0.5)
    adj_hat_csr = degree_for_norm.dot(adj.dot(degree_for_norm))  # D^(-0.5) * A * D^(-0.5)
    adj_hat_coo = adj_hat_csr.tocoo().astype(np.float32)

    # to torch sparse matrix
    indices = torch.from_numpy(np.vstack((adj_hat_coo.row, adj_hat_coo.col)).astype(np.int64))
    values = torch.from_numpy(adj_hat_coo.data)
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(adj_hat_coo.shape))

    return adjacency_matrix, adj_hat_csr


def get_laplacian_matrix(adjacency_matrix_csr: sparse.csr_matrix):
    # since adjacency_matrix_csr is already in form D^(-0.5) * A * D^(-0.5), we can simply get normalized laplacian by:
    laplacian = sparse.eye(adjacency_matrix_csr.shape[0]) - adjacency_matrix_csr
    # rescaling laplacian
    max_eigenval = sparse.linalg.eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0]
    laplacian = 2 * laplacian / max_eigenval - sparse.eye(adjacency_matrix_csr.shape[0])
    # to torch sparse matrix
    laplacian = laplacian.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((laplacian.row, laplacian.col)).astype(np.int64))
    values = torch.from_numpy(laplacian.data)
    laplacian_matrix = torch.sparse_coo_tensor(indices, values, torch.Size(laplacian.shape))
    return laplacian_matrix
