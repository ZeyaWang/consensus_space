# =============================================================================
# Shared manifold for Reliable Clustering Evaluation in Deep Learning
# =============================================================================
# File: towards.py

#
# Description:
# This module implements a method to learn a shared manifold for internal vaildation
# in deep clustering evaluation. Our method can be adapted with different manifold 
# learning method. This implementation is based on TSNE, and extends scikit-learn's 
# TSNE implementation and overrides several key methods in TSNE. 
#
# Based on:
# scikit-learn: https://github.com/scikit-learn/scikit-learn
# TSNE source: sklearn.manifold._t_sne.TSNE
#
# Key Features:
# - Inherits from sklearn.manifold.TSNE
# - Overrides `_fit_transform()`
#
# Usage:
# model = Towards(n_components=2, perplexity=30, my_param=0.5)
# embedding = model.fit_transform(X)
#
# Notes:
# Ensure compatibility with the sklearn version 1.5.1
# This code is for research and prototyping purposes.
#
# =============================================================================




import numpy as np
from scipy.spatial.distance import pdist, squareform
MACHINE_EPSILON = np.finfo(np.double).eps
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_non_negative
from sklearn.manifold import _utils  # type: ignore
from sklearn.manifold import TSNE
# from sklearn.manifold._t_sne import (
#     _gradient_descent,
#     #_joint_probabilities,
#     _kl_divergence,
# )
import scipy.stats as stats


from fusion import weight_fusion_matrices

def _joint_probabilities_towards(distances, desired_perplexity, verbose):
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    return conditional_P


class Towards(TSNE):
    def _fit_prob(self, X):
        if self.learning_rate == "auto":
            self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate

        # X = self._validate_data(
        #     X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
        # )
        if self.metric == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            check_non_negative(
                X,
                (
                    "With metric='precomputed', X "
                    "should contain positive distances."
                ),
            )
        # n_samples = X.shape[0]
        #
        # neighbors_nn = None
        if self.method == "exact":
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("Towards computing pairwise distances...")
                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    # metric_params_ = self.metric_params or {}
                    # distances = pairwise_distances(
                    #     X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    # )
                    raise ValueError("Other metrics are not supported in this version")

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            # if self.metric != "euclidean":
            #     distances **= 2
            # compute the joint probability distribution for the input space
            Pt = _joint_probabilities_towards(distances, self.perplexity, self.verbose)
        else:
            ValueError("method must be exact in this version")
        return Pt

    def _fit_fusion(self, X_list):
        n_samples = X_list[0].shape[0]
        random_state = check_random_state(self.random_state)
        P_list = []

        for X in X_list:
            Pt = self._fit_prob(X)
            P_list.append(Pt)

        if self.method == "exact":
            P, alpha = weight_fusion_matrices(P_list)
            P = P + P.T
            sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
            P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), "All probabilities should be less or then equal to one"
        else:
            #### placeholder for future development of other methods like bh ###
            ValueError("method must be exact at this version")

        X_embedded = 1e-4 * random_state.standard_normal(
            size=(n_samples, self.n_components)
        ).astype(np.float32)
        degrees_of_freedom = max(self.n_components - 1, 1)
        embedding = self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            skip_num_points=0,
        )
        return embedding, P, alpha

    def fit_transform(self, X_list):
        """get the unified manifold embedding
        Parameters
        ----------
        X_list: a list of X; X has {array-like, sparse matrix} of shape (n_samples, n_features)
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        #if self.max_iter is None:
        self._max_iter = 1000
        # else:
        #     self._max_iter = self.max_iter
        for X in X_list:
            self._check_params_vs_input(X)
        embedding, P, alpha = self._fit_fusion(X_list)
        self.embedding_ = embedding
        return self.embedding_, P, alpha




def clustering_accuracy(gtlabels, labels):
    gtlabels = np.array(gtlabels, dtype='int64')
    labels = np.array(labels, dtype='int64')
    cost_matrix = []
    categories = np.unique(gtlabels)
    nr = np.amax(labels) + 1
    for i in np.arange(len(categories)):
      cost_matrix.append(np.bincount(labels[gtlabels == categories[i]], minlength=nr))
    cost_matrix = np.asarray(cost_matrix).T
    row_ind, col_ind = linear_sum_assignment(np.max(cost_matrix) - cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum()) / len(gtlabels)

def calinski_harabasz_score(x,y):
    if len(np.unique(y)) > 1:
        return metrics.calinski_harabasz_score(x,y)
    else:
        return - np.inf

def davies_bouldin_score(x,y):
    if len(np.unique(y)) > 1:
        return - metrics.davies_bouldin_score(x,y)
    else:
        return - np.inf

def silhouette_score(x,y, metric):
    if len(np.unique(y)) > 1:
        return metrics.silhouette_score(x, y, metric=metric)
    else:
        return - np.inf


def clustering_score(x,y, metric):
    if metric == 'dav':
        return davies_bouldin_score(x,y)
    elif metric == 'ch':
        return calinski_harabasz_score(x,y)
    else:
        return silhouette_score(x, y, metric=metric)

def get_files_with_substring_and_suffix(directory, substring, suffix):
    all_files = os.listdir(directory)
    files = [file for file in all_files if substring in file and file.endswith(suffix)]
    return files

def kendalltau(score1, score2):
    '''
    calculate kendal tau
    '''
    stat, pval = stats.kendalltau(score1, score2)
    return np.round(stat, 3), pval

def spearmanr(score1, score2):
    '''
    calculate spearman correlation
    '''
    stat, pval = stats.spearmanr(score1, score2)
    return np.round(stat, 3), pval

if __name__ == '__main__':
    import numpy as np
    from sklearn import metrics
    import os, h5py
    import pickle as pk
    from collections import defaultdict
    import argparse
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster import normalized_mutual_info_score


    modelpath = {
        'jule': 'JULE_hyper',
        'julenum': 'JULE_num',
        'DEPICT': 'DEPICT_hyper',
        'DEPICTnum': 'DEPICT_num',
    }


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FRGC')
    parser.add_argument('--task', default='DEPICT')
    parser.add_argument('--n_components', type=int, default=2, help='number of components')
    parser.add_argument('--perplexity', type=float, default=30.0, help='perplexity')

    args = parser.parse_args()
    eval_data = args.dataset
    task = args.task
    if task == 'jule':
        task_name = 'jule_hyper'
    elif task == 'julenum':
        task_name = 'jule_num'
    else:
        task_name = task
    if not os.path.isdir(task_name):
        os.mkdir(task_name)
    tpath = os.path.join(task_name, 'manifold_{}_{}'.format(args.n_components, args.perplexity))

    if not os.path.isdir(tpath):
        os.mkdir(tpath)

    if 'jule' in task:
        modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'feature'+eval_data, 'h5')
        modelFiles = [m[7:-3] for m in modelFiles]
    else:
        modelFiles = get_files_with_substring_and_suffix(modelpath[task], 'output'+eval_data, 'npz')

    labels = {}
    features = {}
    scored = defaultdict(dict)

    tfname = 'datasets/{}/data4torch.h5'.format(eval_data)
    truth= np.squeeze(np.array(h5py.File(tfname, 'r')['labels']))
    data=np.array(h5py.File(tfname, 'r')['data'])
    data = [data[i].flatten() for i in range(data.shape[0])]
    data = np.stack(data, axis=0)

    if 'jule' in task:
        for m in modelFiles:
            ffname = os.path.join(modelpath[task], 'feature{}.h5'.format(m))
            lfname = os.path.join(modelpath[task], 'label{}.h5'.format(m))
            features[m] = np.array(h5py.File(ffname, 'r')['feature'])
            labels[m] = np.squeeze(np.array(h5py.File(lfname, 'r')['label']))
    else:
        for m in modelFiles:
            files = np.load(os.path.join(modelpath[task], m))
            features[m] = np.array(files['y_features'])
            labels[m] = np.squeeze(np.array(files['y_pred']))

    x_list = []
    for m in modelFiles:
        x = features[m]
        x_list.append(x)

    to_manifold = Towards(n_components=args.n_components, random_state=1, method='exact', perplexity=args.perplexity)
    X_towards, common_P, alpha = to_manifold.fit_transform(x_list)
    kl = to_manifold.kl_divergence_

    print(task, eval_data, data.shape)
    scored['kl'] = kl
    scored['alpha'] = dict(zip(modelFiles, alpha.tolist()))
    scored['common_P'] = common_P
    scored['X_towards'] = X_towards

    davs, chs, euclideans, nmis, accs = [], [], [], [], []
    for i, key in enumerate(modelFiles):
        y = labels[key]
        for metric in ['dav', 'ch', 'euclidean']:
            scored[metric][key] = clustering_score(X_towards, y, metric=metric)
        scored['nmi'][key] = normalized_mutual_info_score(truth, y)
        scored['acc'][key] = clustering_accuracy(truth, y)
        accs.append(scored['acc'][key])
        nmis.append(scored['nmi'][key])
        davs.append(scored['dav'][key])
        chs.append(scored['ch'][key])
        euclideans.append(scored['euclidean'][key])


    print('kendall tau in terms of nmi')
    print(kendalltau(euclideans, nmis))
    print(kendalltau(davs, nmis))
    print(kendalltau(chs, nmis))
    print('kendall tau in terms of acc')
    print(kendalltau(euclideans, accs))
    print(kendalltau(davs, accs))
    print(kendalltau(chs, accs))
    print('spearman rank correlation in terms of nmi')
    print(spearmanr(euclideans, nmis))
    print(spearmanr(davs, nmis))
    print(spearmanr(chs, nmis))
    print('spearman rank correlation in terms of acc')
    print(spearmanr(euclideans, accs))
    print(spearmanr(davs, accs))
    print(spearmanr(chs, accs))


    with open(os.path.join(tpath,'{}_{}_score.pkl'.format(task_name,eval_data)), 'wb') as file:
        pk.dump(scored, file)
