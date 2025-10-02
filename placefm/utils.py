import random

import os
import numpy as np
from shapely.geometry import Point
import torch
import torch.nn.functional as F
import torch.sparse as ts
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from torch_sparse import SparseTensor,fill_diag,matmul,mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils import degree
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import clip_by_rect
from collections import Counter
import folium
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# from deeprobust.graph.utils import *
def is_identity(mx, device):
    identity = torch.eye(mx.size(0), device=device)
    if isinstance(mx, torch.Tensor):
        if is_sparse_tensor(mx):
            dense_tensor = mx.to_dense().float()
        else:
            dense_tensor = mx.float()
    elif isinstance(mx, SparseTensor):
        dense_tensor = mx.to_dense().float()
    else:
        raise ValueError("Input must be a torch.Tensor or torch.sparse.FloatTensor or torch_sparse.SparseTensor")
    return torch.allclose(dense_tensor, identity)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss


def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1)) / n - 0.5)


def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro


def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum() - thresh)
    # return F.relu(adj.sum()-thresh) / n**2


def feature_smoothing(adj, X):
    adj = (adj.t() + adj) / 2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv + 1e-8
    r_inv = r_inv.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat


def row_normalize_tensor(mx):
    mx -= mx.min()
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    k_ss_reg = (k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc


# =================scaling up========#


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def to_camel_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def to_tensor(*vars, **kwargs):
    device = kwargs.get('device', 'cpu')
    tensor_list = []
    for var in vars:
        var = check_type(var, device)
        tensor_list.append(var)
    for key, value in kwargs.items():
        if key != 'device':
            value = check_type(value, device)
            if 'label' in key and len(value.shape) == 1:
                tensor = value.long()
            else:
                tensor = value
            tensor_list.append(tensor)
    if len(tensor_list) == 1:
        return tensor_list[0]
    else:
        return tensor_list


def check_type(var, device):
    if sp.issparse(var):
        var = sparse_mx_to_torch_sparse_tensor(var).coalesce()
    elif isinstance(var, np.ndarray):
        var = torch.from_numpy(var)
    else:
        pass
    return var.float().to(device)


# ============the following is copy from deeprobust/graph/utils.py=================
import scipy.sparse as sp


def encode_onehot(labels):
    """Convert label to onehot format.

    Parameters
    ----------
    labels : numpy.array
        node labels

    Returns
    -------
    numpy.array
        onehot labels
    """
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx


def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.

    Parameters
    ----------
    labels : torch.LongTensor
        node labels

    Returns
    -------
    torch.LongTensor
        onehot labels tensor

    """

    eye = torch.eye(labels.max() + 1).to(labels.device)
    onehot_mx = eye[labels]
    return onehot_mx


def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


def normalize_adj_sgformer(adj):
    """
    Normalize the adjacency matrix. Works for both dense and sparse matrices.

    Args:
    adj (torch.Tensor): The adjacency matrix (either dense or sparse COO).
    device (torch.device): The device to run the normalization on.

    Returns:
    torch.Tensor or SparseTensor: The normalized adjacency matrix.
    """
    # if is_sparse_tensor(adj):
    #     # Sparse matrix normalization
    #     row = adj.indices()[0]
    #     col = adj.indices()[1]
    #     values = adj.values()
    if isinstance(adj, SparseTensor):
        row, col, values = adj.coo()
        # Number of nodes
        N = adj.size(0)

        # Compute degree for normalization
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()

        # Normalize the values directly
        normalized_values = values * d_norm_in * d_norm_out
        normalized_values = torch.nan_to_num(normalized_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Create a new SparseTensor with normalized values
        adj_normalized = SparseTensor(row=row, col=col, value=normalized_values, sparse_sizes=(N, N))

    else:
        # Dense matrix normalization
        N = adj.size(0)

        # Compute degree for normalization
        d = adj.sum(dim=1).float()
        d_norm = torch.diag((1. / d).sqrt())

        # Normalize the dense adjacency matrix
        adj_normalized = d_norm @ adj @ d_norm
        adj_normalized = torch.nan_to_num(adj_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return adj_normalized


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix, return sparse or not
    """
    device = adj.device
    if sparse:
        adj = to_scipy(adj)
        mx = gcn_normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(mx).to(device).coalesce()
        adj = SparseTensor(row=adj.indices()[0], col=adj.indices()[1],
                           value=adj.values(), sparse_sizes=adj.size()).t()
        return adj

    else:
        if len(adj.shape) == 3:
            adjs = []
            for i in range(adj.shape[0]):
                ad = adj[i]
                ad = dense_gcn_norm(ad, device)
                adjs.append(ad)
            return torch.stack(adjs)
        else:
            adj = dense_gcn_norm(adj, device)

            return adj


def dense_gcn_norm(adj, device):
    if type(adj) is not torch.Tensor:
        adj = torch.from_numpy(adj)
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def gcn_normalize_adj(adj, device='cpu'):
    if sp.issparse(adj):
        return sparse_gcn_norm(adj)
    elif type(adj) is torch.Tensor:
        return dense_gcn_norm(adj, device)
    else:
        return dense_gcn_norm(adj, device).numpy()


def sparse_gcn_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj).dot(r_mat_inv)
    return adj


def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    # mx = mx.dot(r_mat_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def degree_normalize_sparse_tensor(adj, fill_value=1):
    """degree_normalize_sparse_tensor.
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add



    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight
    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def degree_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """

    device = adj.device
    if sparse:
        # return  degree_normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = degree_normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def roc_auc(output, labels, is_sigmoid=False):
    """Return ROC-AUC score of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        true labels (0 or 1)
    is_sigmoid : bool, optional
        If True, apply sigmoid thresholding on the output, by default False.

    Returns
    -------
    float
        ROC-AUC score
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    labels = labels.cpu().numpy()
    output = output.cpu().numpy()

    if not is_sigmoid:
        # For multi-class classification (softmax output), get probabilities for the positive class (class 1).
        if output.shape[1] > 1:
            output = output[:, 1]  # Use the probabilities of the positive class
        else:
            output = np.argmax(output, axis=1)
    else:
        # For binary classification (sigmoid output)
        output = output[:, 1]

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(labels, output)

    return roc_auc


def f1_macro(output, labels, is_sigmoid=False):
    """Return F1-macro score of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        true labels (0 or 1)

    Returns
    -------
    float
        F1-macro score
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)

    labels = labels.cpu().numpy()
    output = output.cpu().numpy()

    if not is_sigmoid:
        output = np.argmax(output, axis=1)
    else:
        output = output[:, 1]
        output[output > 0.5] = 1
        output[output <= 0.5] = 0

    f1 = f1_score(labels, output, average="macro")

    return f1


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse_coo_tensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


# slower version....
# sparse_mx = sparse_mx.tocoo().astype(np.float32)
# indices = torch.from_numpy(
#     np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
# values = torch.from_numpy(sparse_mx.data)
# shape = torch.Size(sparse_mx.shape)
# return torch.sparse.FloatTensor(indices, values, shape)


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def get_train_test(nnodes, test_size=0.8, stratify=None, seed=None):
    """This function returns training and test set without validation.
    It can be used for settings of different label rates.

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_test :
        node test indices
    """
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - test_size
    idx_train, idx_test = train_test_split(idx, random_state=None,
                                           train_size=train_size,
                                           test_size=test_size,
                                           stratify=stratify)

    return idx_train, idx_test


def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20:])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]
    return idx_train, idx_val, idx_test


def get_train_test_labelrate(labels, label_rate):
    """Get train test according to given label rate.
    """
    nclass = labels.max() + 1
    train_size = int(round(len(labels) * label_rate / nclass))
    print("=== train_size = %s ===" % train_size)
    idx_train, idx_val, idx_test = get_splits_each_class(labels, train_size=train_size)
    return idx_train, idx_test


def get_splits_each_class(labels, train_size):
    """We randomly sample n instances for class, where n = train_size.
    """
    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: train_size])).astype(np.int)
        idx_val = np.hstack((idx_val, labels_i[train_size: 2 * train_size])).astype(np.int)
        idx_test = np.hstack((idx_test, labels_i[2 * train_size:])).astype(np.int)

    return np.random.permutation(idx_train), np.random.permutation(idx_val), \
        np.random.permutation(idx_test)


def unravel_index(index, array_shape):
    rows = torch.div(index, array_shape[1], rounding_mode='trunc')
    cols = index % array_shape[1]
    return rows, cols


def get_degree_squence(adj):
    try:
        return adj.sum(0)
    except:
        return ts.sum(adj, dim=1).to_dense()


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    """ Adopted from https://github.com/danielzuegner/nettack
    """
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                            + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll


def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.

    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.

    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]


# def visualize(your_var):
#     """visualize computation graph"""
#     from torchviz import make_dot
#     make_dot(your_var).view()
#

def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)


def add_mask(data, dataset):
    """data: ogb-arxiv pyg data format"""
    # for arxiv
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    n = data.x.shape[0]
    data.train_mask = index_to_mask(train_idx, n)
    data.val_mask = index_to_mask(valid_idx, n)
    data.test_mask = index_to_mask(test_idx, n)
    data.y = data.y.squeeze()
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def add_feature_noise(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    # noise = torch.normal(mean=torch.zeros(int(noise_ratio*n), d), std=1)
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio * n), d))).to(data.x.device)
    indices = np.arange(n)
    indices = np.random.permutation(indices)[: int(noise_ratio * n)]
    delta_feat = torch.zeros_like(data.x)
    delta_feat[indices] = noise - data.x[indices]
    data.x[indices] = noise
    mask = np.zeros(n)
    mask[indices] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask


def add_feature_noise_test(data, noise_ratio, seed):
    np.random.seed(seed)
    n, d = data.x.shape
    indices = np.arange(n)
    test_nodes = indices[data.test_mask.cpu()]
    selected = np.random.permutation(test_nodes)[: int(noise_ratio * len(test_nodes))]
    noise = torch.FloatTensor(np.random.normal(0, 1, size=(int(noise_ratio * len(test_nodes)), d)))
    noise = noise.to(data.x.device)

    delta_feat = torch.zeros_like(data.x)
    delta_feat[selected] = noise - data.x[selected]
    data.x[selected] = noise
    # mask = np.zeros(len(test_nodes))
    mask = np.zeros(n)
    mask[selected] = 1
    mask = torch.tensor(mask).bool().to(data.x.device)
    return delta_feat, mask


def estimate_eps(X, min_samples=5):
    """ 
    
    Estimate the epsilon parameter for DBSCAN clustering using the k-th nearest neighbor distance method.
    
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)

    # sort the k-th NN distance
    distances = np.sort(distances[:, -1])
    
    # find "elbow"
    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
    eps = distances[kneedle.knee] if kneedle.knee is not None else np.median(distances)

    # Plot the knee curve and save the figure
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(len(distances)), distances, label="k-th NN Distance")
    # if kneedle.knee is not None:
    #     plt.axvline(x=kneedle.knee, color='r', linestyle='--', label="Knee Point")
    # plt.xlabel("Points")
    # plt.ylabel("Distance")
    # plt.title("Knee Curve for Epsilon Estimation")
    # plt.legend()
    # plt.savefig("knee_curve.png", dpi=300)
    # plt.close()

    return eps


def filter_pois_for_visualization(feats, lat_lon, categories, cluster_ids, edge_index):
    distances = torch.cdist(lat_lon, lat_lon, p=2)  # Compute pairwise distances
    mean_distances = distances.mean(dim=1)  # Compute mean distance for each POI
    threshold = mean_distances.mean() + 2 * mean_distances.std()  # Define a threshold (mean + 2*std)
    valid_pois_mask = mean_distances <= threshold  # Mask for valid POIs
    lat_lon = lat_lon[valid_pois_mask]
    print(f"VISUALIZATION: Filtered {len(feats) - valid_pois_mask.sum().item()} out of {len(feats)} POIs")
    feats = feats[valid_pois_mask]
    categories = [categories[i] for i in range(len(categories)) if valid_pois_mask[i]]
    cluster_ids = cluster_ids[valid_pois_mask]
    # Remove edges from edge_index related to filtered POIs
    valid_indices = torch.nonzero(valid_pois_mask).squeeze()
    index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}
    filtered_edges = []
    for edge in edge_index.T:
        if edge[0].item() in index_map and edge[1].item() in index_map:
            filtered_edges.append([index_map[edge[0].item()], index_map[edge[1].item()]])
    # Keep only edges that were NOT filtered out
    filtered_edges_set = set(tuple(edge) for edge in filtered_edges)
    all_edges = [tuple(edge.tolist()) for edge in edge_index.T]
    remaining_edges = [edge for edge in all_edges if edge not in filtered_edges_set]
    edge_index = torch.tensor(remaining_edges, dtype=torch.long).T

    return feats, lat_lon, categories, cluster_ids, edge_index


def plot_tsne_pois(args, feats, categories, cluster_ids, region_id):

    


    feats_np = feats.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=args.seed)
    feats_2d = tsne.fit_transform(feats_np)

    # Plot the clustering results
    plt.figure(figsize=(10, 8))
    # feats_2d = feats_np[:, :2]  # Assuming the first two dimensions for 2D visualization

    unique_labels = set(cluster_ids)
    # Plot clusters
    for label in unique_labels:
        cluster_points = feats_2d[cluster_ids == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.7)

    # Highlight centroids
    centroids_2d = np.array([feats_2d[cluster_ids == label].mean(axis=0) for label in unique_labels if label != -1])
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100, label='Centroids')
    # Annotate each point with its category
    for i, (x, y) in enumerate(feats_2d):
        # annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
        annotation = f"{i}"
        plt.text(x, y, annotation, fontsize=8, ha='center', va='center')

    plt.title(f't-SNE Clustering Results for region {region_id}')
    plt.legend()
    plt.grid(True)

    # Save the figure
    if os.path.exists(f"../checkpoints/logs/placefm/{args.state}") is False:
        os.mkdir(f"../checkpoints/logs/placefm/{args.state}")
    save_path = f"../checkpoints/logs/placefm/{args.state}/tsne_region_{region_id}_{args.clustering_method}.png"
    plt.savefig(save_path)
    plt.close()

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct finite Voronoi regions.
    Source: https://stackoverflow.com/a/20678647
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()*2

    # Construct a map of ridges for each point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(int(p1), []).append((int(p2), v1, v2))
        all_ridges.setdefault(int(p2), []).append((int(p1), v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        if p1 not in all_ridges:
            new_regions.append([v for v in vertices if v >= 0])
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def plot_spatial_pois(args, lat_lon, categories, edge_index, cluster_ids, region_id):
    

    plt.figure(figsize=(25, 22))
    longitudes = lat_lon[:, 1]  
    latitudes = lat_lon[:, 0]   

    # Create Voronoi diagram
    points = np.vstack([longitudes, latitudes]).T
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    

    # Bounding box (clip polygons here)
    min_x, max_x = longitudes.min().item(), longitudes.max().item()
    min_y, max_y = latitudes.min().item(), latitudes.max().item()

    # Generate distinct colors per cluster
    unique_cluster_ids = set(cluster_ids)
    color_palette = plt.cm.get_cmap('hsv', len(unique_cluster_ids))  # HSV ensures distinct colors for many clusters

    colors = {cid: color_palette(i) for i, cid in enumerate(unique_cluster_ids)}

    # Scatter POIs
    for idx in range(len(regions)):
        region = regions[idx]
        cid = cluster_ids[idx] 

        polygon = vertices[region]
        poly = Polygon(polygon)
        poly = clip_by_rect(poly, min_x, min_y, max_x, max_y)
        if isinstance(poly, MultiPolygon):
            print(f"region {idx} is a MultiPolygon, skipping visualization for this region.")
            # poly = max(poly.geoms, key=lambda p: p.area)  # Select the largest polygon
            continue
        
        x, y = poly.exterior.xy

        # Check for overlapping regions
        for j in range(len(regions)):
            other_region = regions[j]
            other_polygon = vertices[other_region]
            other_poly = Polygon(other_polygon)

            overlap = False
            if j != idx:
                poly = poly.buffer(0)  # Attempt to fix invalid geometry
                other_poly = other_poly.buffer(0)  # Attempt to fix invalid geometry
                if poly.is_valid and other_poly.is_valid and poly.intersection(other_poly).area > 0.0:
                    overlap = True
                    print(f"Overlap detected between region {idx} and region {j}")
        if overlap is False:
            plt.fill(x, y, alpha=0.3, color=colors[cid], edgecolor=colors[cid], linewidth=0.0)

            # Plot the corresponding POI for this region
            plt.scatter(longitudes[idx], latitudes[idx], c=[colors[cid]], s=20, alpha=1.0)

    

    # Annotate each POI
    for i in range(len(lat_lon)):
        annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
        plt.text(longitudes[i], latitudes[i], annotation, fontsize=8,
                 ha='center', va='center')

    # Plot edges if provided
    if edge_index is not None:
        for edge in edge_index.T:
            start, end = edge
            plt.plot(
                [longitudes[start], longitudes[end]],
                [latitudes[start], latitudes[end]],
                color='gray', alpha=0.5, linewidth=0.5
            )

    plt.title(f'Voronoi Clustered Regions for region {region_id} clustered into {len(unique_cluster_ids)} places')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])
    plt.grid(True)

    # Save
    save_dir = f"../checkpoints/logs/placefm/{args.state}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/voronoi_region_{region_id}_{args.clustering_method}.png"
    plt.savefig(save_path)
    plt.close()


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct finite Voronoi regions.
    Source: https://stackoverflow.com/a/20678647
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()*2

    # Construct a map of ridges for each point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(int(p1), []).append((int(p2), v1, v2))
        all_ridges.setdefault(int(p2), []).append((int(p1), v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        if p1 not in all_ridges:
            new_regions.append([v for v in vertices if v >= 0])
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def plot_spatial_pois_folium(args, lat_lon, categories, edge_index, edge_weight, cluster_ids, region_id, cluster_color_map):
    """
    Interactive Folium map version of Voronoi POI plotting.
    """

    

    longitudes = lat_lon[:, 1]
    latitudes = lat_lon[:, 0]

    # Initialize folium map
    center_lat, center_lon = latitudes.mean().item(), longitudes.mean().item()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron', control_scale=True)
    
    region_zip = str(region_id)

    points = np.vstack([longitudes, latitudes]).T
    # vor = Voronoi(points)
    # regions, vertices = voronoi_finite_polygons_2d(vor)

    # Bounding box (clip polygons here)
    # min_x, max_x = longitudes.min().item(), longitudes.max().item()
    # min_y, max_y = latitudes.min().item(), latitudes.max().item()


    # US Census Bureau TIGER/Line Shapefiles for ZIP Code Tabulation Areas (ZCTA)
    # Direct URL to GeoJSON (for 2020): https://www2.census.gov/geo/tiger/TIGER2020/ZCTA5/tl_2020_us_zcta520.geojson
    shapefile_path = "../data/fsq/Census/tl_2024_us_zcta520/tl_2024_us_zcta520.shp"
    gdf = gpd.read_file(shapefile_path)
    region_gdf = gdf[gdf['ZCTA5CE20'] == region_zip]
    if not region_gdf.empty:
        for _, row in region_gdf.iterrows():
            boundary = row.geometry
            if boundary.geom_type == 'Polygon':
                coords = [(y, x) for x, y in boundary.exterior.coords]
                folium.PolyLine(
                    locations=coords,
                    color='red',
                    weight=3,
                    opacity=0.8,
                    dash_array='10,5',
                    tooltip=f'Zipcode {region_zip} Boundary'
                ).add_to(m)
            elif boundary.geom_type == 'MultiPolygon':
                for poly in boundary.geoms:
                    coords = [(y, x) for x, y in poly.exterior.coords]
                    folium.PolyLine(
                        locations=coords,
                        color='red',
                        weight=3,
                        opacity=0.8,
                        dash_array='10,5',
                        tooltip=f'Zipcode {region_zip} Boundary'
                    ).add_to(m)
    else:
        print(f"Zipcode {region_zip} not found in census data.")


    # filter out points outside the boundary
    boundary = region_gdf.geometry.values[0]
    filtered_points = []
    filtered_lat_lon = []
    filtered_cluster_ids = []
    filtered_categories = []
    kept_indices = []
    for i in range(len(lat_lon)):
        pt = Point(torch.tensor([lat_lon[i][1], lat_lon[i][0]]))
        if boundary.contains(pt):
            filtered_points.append(points[i])
            filtered_lat_lon.append(lat_lon[i])
            filtered_cluster_ids.append(cluster_ids[i])
            filtered_categories.append(categories[i])
            kept_indices.append(i)
            
    
            
    points = np.array(filtered_points)
    lat_lon = np.array(filtered_lat_lon)
    cluster_ids = np.array(filtered_cluster_ids)
    categories = np.array(filtered_categories)
    latitudes = lat_lon[:, 0]
    longitudes = lat_lon[:, 1]

    # Extract edges inside the the region
    region_edge_index = []

    poi_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
    # Mask for edges where both src and dst are in idx
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    idx_set = set(kept_indices)
    mask = torch.tensor([(src.item() in idx_set and dst.item() in idx_set) for src, dst in zip(src_nodes, dst_nodes)], dtype=torch.bool)

    # Filter edges
    filtered_src = src_nodes[mask]
    filtered_dst = dst_nodes[mask]
    region_edge_weight = edge_weight[mask]

    # Map old indices to new indices
    mapped_src = torch.tensor([poi_idx_map[src.item()] for src in filtered_src], dtype=torch.long)
    mapped_dst = torch.tensor([poi_idx_map[dst.item()] for dst in filtered_dst], dtype=torch.long)

    region_edge_index = torch.stack([mapped_src, mapped_dst], dim=0)

    # region_edge_index = torch.tensor(region_edge_index, dtype=torch.long).t().contiguous()


    print(f"After filtering, {len(points)} POIs remain within zipcode {region_zip}")


    # Create Voronoi diagram
    regions, region_pts = voronoi_regions_from_coords(points, boundary)


    # Generate distinct colors per cluster
    # unique_cluster_ids = set(cluster_ids)
    # color_palette = plt.cm.get_cmap('hsv', len(unique_cluster_ids))
    # colors = {cid: f"rgba({int(255*r)}, {int(255*g)}, {int(255*b)}, 0.5)"
    #           for cid, (r, g, b, _) in zip(unique_cluster_ids, color_palette(range(len(unique_cluster_ids))))}
    colors = cluster_color_map
    colors = {cid: f"rgba({int(255*r)}, {int(255*g)}, {int(255*b)}, 0.5)"
              for cid, (r, g, b) in cluster_color_map.items()}
    

    # Add Voronoi polygons
    # Calculate the cluster sizes
    cluster_sizes = Counter(cluster_ids)

    # Get the top 5 clusters with the most members
    top_clusters = cluster_sizes.most_common(10)
    top_cids = [cid for cid, _ in top_clusters]

    print(f"Top 5 clusters with the most members: {top_clusters}")
    for idx in range(len(regions)):
        region_pt = region_pts[idx]
        poly = regions[idx]
        cid = cluster_ids[region_pt[0]]
        if cid not in top_cids:
            continue  
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:
                folium.Polygon(
                    locations=[(y, x) for x, y in sub_poly.exterior.coords],
                    color=colors[cid],
                    fill=True,
                    fill_color=colors[cid],
                    fill_opacity=0.8,
                    weight=0,
                ).add_to(m)

        elif isinstance(poly, Polygon):
            folium.Polygon(
                locations=[(y, x) for x, y in poly.exterior.coords],
                color=colors[cid],
                fill=True,
                fill_color=colors[cid],
                fill_opacity=0.8,
                weight=0,
            ).add_to(m)

        
    # Plot POIs as markers
    for idx in range(len(lat_lon)):


        url = "/scratch/mhashe4/repos/PlaceFM/placefm/plots/icons/{}".format
        if categories[idx] is not None and str(categories[idx]) != 'nan':
            category = str(categories[idx]).split('>')[0].strip().lower()
            if category == "community and government":
                icon = "government"
            elif category == "Business and professional services".lower():
                icon = "briefcase"
            elif category == "Retail".lower():
                icon = "shopping-cart"
            elif category == "Dining and drinking".lower():
                icon = "food"
            elif category == "arts and entertainment".lower():
                icon = "art"
            elif category == "landmarks and outdoors".lower():
                icon = "star"
            elif category == "Sports and recreation".lower():
                icon = "gym"
            elif category == "health and medicine".lower():
                icon = "heart"
            elif category == "travel and transportation".lower():
                icon = "bus"
            elif category == "event".lower():
                icon = "event"
            else:
                icon = category.replace(' ', '_').replace('/', '_')
            icon_url = url(f"{icon}.png")

        
        if cluster_ids[idx] in top_cids:
            icon = folium.CustomIcon(
                icon_url,
                icon_size=(20, 20),
            )

            folium.Marker(
                location=(latitudes[idx], longitudes[idx]),
                radius=3,
                color=colors[cluster_ids[idx]],
                fill=True,
                fill_color=colors[cluster_ids[idx]],
                fill_opacity=0.5,
                tooltip=f"POI {idx}: {str(categories[idx])} [Place {cluster_ids[idx]}]",
                icon=icon
            ).add_to(m)




    # Add edges if provided
    # Normalize edge weights to be scaled to (0, 1)
    if region_edge_weight is not None:
        min_weight = region_edge_weight.min()
        max_weight = region_edge_weight.max()
        region_edge_weight = (region_edge_weight - min_weight) / (max_weight - min_weight + 1e-8)

    # Remove edges with weights below the threshold
    # weight_threshold = 0.4  # Define your threshold here
    # valid_edge_mask = region_edge_weight >= weight_threshold
    # region_edge_index = region_edge_index[:, valid_edge_mask]
    # valid_edge_mask = valid_edge_mask.to(region_edge_weight.device)
    # region_edge_weight = region_edge_weight[valid_edge_mask]

    if region_edge_index is not None:
        for i, edge in enumerate(region_edge_index.T):
            start, end = edge
            folium.PolyLine(
                locations=[
                    (latitudes[start], longitudes[start]),
                    (latitudes[end], longitudes[end])
                ],
                color="gray",
                weight=region_edge_weight[i].item(),
                opacity=region_edge_weight[i].item(),
                # tooltip=f"Weight: {region_edge_weight[i].item():.4f}"
            ).add_to(m)

    # Save interactive map
    save_dir = f"../checkpoints/logs/placefm/{args.state}/figs/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/voronoi_region_{region_id}_{args.clustering_method}_g{args.placefm_agg_gamma}_a{args.placefm_agg_alpha}_b{args.placefm_agg_beta}_rr{args.placefm_kmeans_reduction_ratio}_prop{~args.no_prop}.html"
    m.save(save_path)

    print(f"Interactive map saved to {save_path}")



def plot_spatial_pois(args, lat_lon, categories, edge_index, cluster_ids, region_id):
    

    plt.figure(figsize=(25, 22))
    longitudes = lat_lon[:, 1]  
    latitudes = lat_lon[:, 0]   

    # Create Voronoi diagram
    points = np.vstack([longitudes, latitudes]).T
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    

    # Bounding box (clip polygons here)
    min_x, max_x = longitudes.min().item(), longitudes.max().item()
    min_y, max_y = latitudes.min().item(), latitudes.max().item()

    # Generate distinct colors per cluster
    unique_cluster_ids = set(cluster_ids)
    color_palette = plt.cm.get_cmap('hsv', len(unique_cluster_ids))  # HSV ensures distinct colors for many clusters

    colors = {cid: color_palette(i) for i, cid in enumerate(unique_cluster_ids)}

    # Scatter POIs
    for idx in range(len(regions)):
        region = regions[idx]
        cid = cluster_ids[idx] 

        polygon = vertices[region]
        poly = Polygon(polygon)
        poly = clip_by_rect(poly, min_x, min_y, max_x, max_y)
        if isinstance(poly, MultiPolygon):
            print(f"region {idx} is a MultiPolygon, skipping visualization for this region.")
            # poly = max(poly.geoms, key=lambda p: p.area)  # Select the largest polygon
            continue
        
        x, y = poly.exterior.xy

        # Check for overlapping regions
        for j in range(len(regions)):
            other_region = regions[j]
            other_polygon = vertices[other_region]
            other_poly = Polygon(other_polygon)

            overlap = False
            if j != idx:
                poly = poly.buffer(0)  # Attempt to fix invalid geometry
                other_poly = other_poly.buffer(0)  # Attempt to fix invalid geometry
                if poly.is_valid and other_poly.is_valid and poly.intersection(other_poly).area > 0.0:
                    overlap = True
                    print(f"Overlap detected between region {idx} and region {j}")
        if overlap is False:
            plt.fill(x, y, alpha=0.3, color=colors[cid], edgecolor=colors[cid], linewidth=0.0)

            # Plot the corresponding POI for this region
            plt.scatter(longitudes[idx], latitudes[idx], c=[colors[cid]], s=20, alpha=1.0)

    

    # Annotate each POI
    for i in range(len(lat_lon)):
        annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
        plt.text(longitudes[i], latitudes[i], annotation, fontsize=8,
                 ha='center', va='center')

    # Plot edges if provided
    if edge_index is not None:
        for edge in edge_index.T:
            start, end = edge
            plt.plot(
                [longitudes[start], longitudes[end]],
                [latitudes[start], latitudes[end]],
                color='gray', alpha=0.5, linewidth=0.5
            )

    plt.title(f'Voronoi Clustered Regions for region {region_id} clustered into {len(unique_cluster_ids)} places')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(vor.min_bound[0], vor.max_bound[0])
    plt.ylim(vor.min_bound[1], vor.max_bound[1])
    plt.grid(True)

    # Save
    save_dir = f"../checkpoints/logs/placefm/{args.state}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/voronoi_region_{region_id}_{args.clustering_method}.png"
    plt.savefig(save_path)
    plt.close()

def assign_cluster_colors(feats, categories, cluster_ids, num_colors=8, method='gradient'):
    """
    Assigns colors to clusters using either random coloring or gradient coloring based on feature similarity.
    Also generates a color palette plot with semantic descriptions (top-k common categories per color).

    Args:
        feats: Tensor of shape [N, D], feature vectors for each point.
        categories: List or array of category labels for each point.
        cluster_ids: Array of cluster assignments for each point.
        num_colors: Number of base colors for gradient coloring.
        method: 'random' or 'gradient'.

    Returns:
        cluster_color_map: Dict mapping cluster_id to color (RGB tuple).
        color_semantics: Dict mapping color index to top-k categories.
    """
    import matplotlib.pyplot as plt

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    feats_np = feats.cpu().numpy() if torch.is_tensor(feats) else np.array(feats)
    cluster_centroids = np.array([feats_np[cluster_ids == cid].mean(axis=0) for cid in unique_clusters])

    if method == 'random':
        colors = plt.cm.get_cmap('tab20', n_clusters)
        cluster_color_map = {cid: colors(i % n_clusters)[:3] for i, cid in enumerate(unique_clusters)}
        color_indices = {cid: i for i, cid in enumerate(unique_clusters)}
    else:
        # Gradient coloring: select num_colors base clusters, assign colors, interpolate for others
        base_indices = np.linspace(0, n_clusters - 1, num_colors, dtype=int)
        base_cids = unique_clusters[base_indices]
        color_palette = sns.color_palette('crest', num_colors)
        # base_colors = [np.array(color_palette(i)[:3]) for i in range(num_colors)]
        base_colors = color_palette 
        base_centroids = cluster_centroids[base_indices]

        # Assign each cluster to nearest base centroid, then interpolate color
        cluster_color_map = {}
        color_indices = {}
        for i, cid in enumerate(unique_clusters):
            dists = np.linalg.norm(base_centroids - cluster_centroids[i], axis=1)
            closest = np.argmin(dists)
            # Interpolate color based on distance to closest base centroid
            interp = 1 - (dists[closest] / (dists.max() + 1e-8))
            base_color = np.array(base_colors[closest])
            color = base_color * interp + np.array([1, 1, 1]) * (1 - interp)
            cluster_color_map[cid] = tuple(color)
            color_indices[cid] = closest

    # Semantic description: top-k categories per color
    # k = 5
    # color_semantics = {}
    # for color_idx in range(num_colors):
    #     cids = [cid for cid, idx in color_indices.items() if idx == color_idx]
    #     cat_list = []
    #     for cid in cids:
    #         cat_list.extend(categories[cluster_ids == cid])
    #     topk = [cat for cat, _ in Counter(cat_list).most_common(k)]
    #     color_semantics[color_idx] = topk


    return cluster_color_map





    # plt.figure(figsize=(25, 22))  
    # longitudes = lat_lon[:, 0]  # Longitude
    # latitudes = lat_lon[:, 1]  # Latitude

    # unique_labels = np.unique(cluster_ids)

    # # --- Draw each cluster with convex hull ---
    # for label in unique_labels:
    #     mask = cluster_ids == label
    #     cluster_points = lat_lon[mask]

    #     # Plot points of this cluster
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.7, label=f'Cluster {label}')

    #     Draw convex hull if enough points
    #     if cluster_points.shape[0] >= 3:
    #         hull = ConvexHull(cluster_points)
    #         hull_points = cluster_points[hull.vertices]

    #         # Fill polygon
    #         plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.1)

    #         # Draw hull edges
    #         for simplex in hull.simplices:
    #             plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], "k-", linewidth=0.1)

        

    
    # # --- Annotate each point with category ---
    # for i in range(len(lat_lon)):
    #     annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
    #     plt.text(longitudes[i], latitudes[i], annotation, fontsize=8, ha='center', va='center')

    # # --- Plot edges if provided ---
    # if edge_index is not None:
    #     for edge in edge_index.T:
    #         start, end = edge
    #         plt.plot(
    #             [longitudes[start], longitudes[end]],
    #             [latitudes[start], latitudes[end]],
    #             color='gray', alpha=0.6, linewidth=0.5
    #         )

    # # --- Formatting ---
    # plt.title(f'Geographical Clustering Results for region {region_id}')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()
    # plt.grid(True)

    # # --- Save ---
    # save_dir = f"../checkpoints/logs/placefm/{args.city}"
    # os.makedirs(save_dir, exist_ok=True)
    # geo_save_path = os.path.join(save_dir, f"region_{region_id}.png")
    # plt.savefig(geo_save_path)
    # plt.close()


def plot_absolute_error(args, state, zipcodes):

    # Load ZIP Code Tabulation Areas (ZCTAs)
    zcta_shapefile_path = "../data/fsq/Census/tl_2024_us_zcta520/tl_2024_us_zcta520.shp"
    state_shapefile_path = "/scratch/mhashe4/repos/fm/data/cb_2024_us_state_20m/cb_2024_us_state_20m.shp"
    
    gdf_state = gpd.read_file(state_shapefile_path)
    gdf_state = gdf_state[gdf_state['STUSPS'] == state]

    gdf_zcta = gpd.read_file(zcta_shapefile_path)
    


    
    region_gdf = gdf_zcta[gdf_zcta['ZCTA5CE20'].isin(zipcodes.keys())]



    # Generate a random color for each ZIP code and plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the state boundary
    gdf_state.boundary.plot(ax=ax, color='black', linewidth=1)

    # Normalize the values in the zipcodes dictionary to create a gradient
    values = list(zipcodes.values())
    min_value = min(values)
    max_value = max(values)

    # Create 4 groups of darkness based on normalized values
    thresholds = np.linspace(min_value, max_value, 30)  # 4 groups, 10 thresholds
    palette = sns.color_palette("rocket", n_colors=5)
    palette.reverse()

    for _, row in region_gdf.iterrows():
        value = zipcodes[row['ZCTA5CE20']]

        if thresholds[0] <= value < thresholds[2]:
            color = palette[0]  # Lightest color
        elif thresholds[2] <= value < thresholds[8]:
            color = palette[1]
        elif thresholds[8] <= value < thresholds[15]:
            color = palette[2]
        elif thresholds[15] <= value < thresholds[20]:
            color = palette[3]
        else:
            color = palette[4]    # Darkest color
        row_geometry = row['geometry']
        if row_geometry is not None:
            gpd.GeoSeries([row_geometry]).plot(ax=ax, color=color, edgecolor='black', linewidth=0.5)
        if row_geometry is not None:
            gpd.GeoSeries([row_geometry]).plot(ax=ax, color=color, edgecolor='black', linewidth=0.5)


    # Add a legend for the color thresholds

    legend_elements = []
    for i, threshold in enumerate(thresholds):
        if i in [2, 8, 15, 20, 29]:
            label = f"<= {thresholds[i] / 1000:.0f}"
            if i == 2:
                palette_index = 0
            elif i == 8:
                palette_index = 1
            elif i == 15:
                palette_index = 2
            elif i == 20:
                palette_index = 3       
            else:
                palette_index = 4
            legend_elements.append(Patch(facecolor=palette[palette_index], edgecolor='black', label=label))

    legend_elements.append(Patch(facecolor="white", edgecolor='black', label="No data"))

    plt.legend(handles=legend_elements, title="Absolute Error Ranges", loc="lower right", fontsize=8, title_fontsize=10)

    # Add a title indicating the state
    plt.title(f"Absolute Error Visualization for ZIP code regions in state: {state}", fontsize=14, fontweight="bold")


    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"plots/{args.state}_abs_err.png")  # Save the figure