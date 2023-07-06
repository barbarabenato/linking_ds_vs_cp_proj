from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis
import umap
import sys


def reduce_to_2d(dim_red_name, X):
    
    if dim_red_name == 'tsne':
        X_embedded = TSNE(n_components=2, method="exact").fit_transform(X)
    elif dim_red_name == 'umap':
        X_embedded = umap.UMAP().fit_transform(X)
    elif dim_red_name == 'pca':
        X_embedded = PCA(n_components=2).fit_transform(X)
    elif dim_red_name == 'kpca':
        X_embedded = KernelPCA(n_components=2).fit_transform(X)
    elif dim_red_name == 'fica':
        X_embedded = FastICA(n_components=2).fit_transform(X)
    elif dim_red_name == 'isomap':
        X_embedded = Isomap(n_components=2).fit_transform(X)
    elif dim_red_name == 'fa':
        X_embedded = FactorAnalysis(n_components=2).fit_transform(X)
    elif dim_red_name == 'lle':
        X_embedded = LocallyLinearEmbedding(n_components=2).fit_transform(X)
    elif dim_red_name == 'mlle':
        X_embedded = LocallyLinearEmbedding(n_components=2,method='modified').fit_transform(X)
    elif dim_red_name == 'mds':
        X_embedded = MDS(n_components=2).fit_transform(X)
    else:
        print("None dimensionality techniques chosen.")
        sys.exit(0)

    X_embedded = X_embedded.astype("float32")

    return X_embedded



