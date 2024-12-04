import os
import torch
import random
import numpy as np
import scanpy as sc
from anndata import AnnData
from torch.backends import cudnn


def leiden_cluster(result_embedding, spatial_coords):

    X = np.load(result_embedding)
    adata = AnnData(X)
    adata.obsm['spatial'] = spatial_coords

    sc.pp.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1)
    sc.pl.spatial(adata, color='leiden', spot_size=1)


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
