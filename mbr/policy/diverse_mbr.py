import numpy as np
from sklearn_extra.cluster import KMedoids


def generate_objective(k, div_pen, matrix):

    def objective(X: np.array):
        n = matrix.shape[0]
        matrix_ = np.copy(matrix)
        np.fill_diagonal(matrix, 0.0)  # Diagonals are not used in the objective.
        # The normalization is applied to make div_pen be in a range of (0, 1).
        # similarity has (n - 1) * k comparisons to give the score, so normalized so.
        # diversity has k * (k - 1) comparisons to give the score, so normalized so.

        # The larger the score is, the more similar it is to the distribution.
        similarity = np.ones((1, n)) @ matrix_ @ X / (n - 1) / k
        # The smaller the score is, the larger the diversity is.
        diveristy = -np.transpose(X) @ matrix_ @ X * div_pen / k / (k - 1)

        return similarity + diveristy

    return objective


def gbfs(func, n, k):
    # Because the problem is a submodular function optimization,
    # greedy search is guaranteed of (1-1/e)-approximation.
    # Empirically it is close to optimal.

    node = np.zeros(n, dtype=int)

    for nsize in range(k):

        # TODO: This procedure can be refactored with lambda
        cur_best, cur_best_i = -np.inf, -1
        for i in range(n):
            if node[i] == 1:
                continue
            next_node = np.copy(node)
            next_node[i] = 1
            obj = func(next_node)

            if obj > cur_best:
                cur_best = obj
                cur_best_i = i
        node[cur_best_i] = 1

    return node


def local_search(func, init, iterations=100, neighbor=2):
    n = init.shape[0]
    k = sum(init)

    node = np.copy(init)

    for i in range(iterations):
        cur_node = np.copy(node)
        neighbor = neighbor
        indices = np.where(node == 1)[0]
        removed_cand = np.random.choice(k, neighbor, replace=False)

        for l in range(neighbor):
            node[indices[removed_cand[l]]] = 0

        for l in range(neighbor):
            cur_best, cur_best_i = -np.inf, -1
            for i in range(n):
                if node[i] == 1:
                    continue
                next_node = np.copy(node)
                next_node[i] = 1
                obj = func(next_node)

                if obj > cur_best:
                    cur_best = obj
                    cur_best_i = i
            node[cur_best_i] = 1

        if func(node) < func(cur_node):
            node = cur_node

    return node


def compute_dmbr(
    hyp=None, score_function=None, matrix=None, weights=None, src=None, k=1, div_pen=0.0
):
    # TODO: Weights are not applied in current implementation.
    assert (score_function is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, score_function, [src] * len(hyp))

    n = matrix.shape[0]
    obj = generate_objective(k=k, div_pen=div_pen, matrix=matrix)
    gbfs_result = gbfs(obj, n, k=k)
    local_result = local_search(obj, gbfs_result, iterations=20, neighbor=1)

    k_bests = np.where(local_result == 1)[0]
    return k_bests


def compute_kmedmbr(
    hyp=None, score_function=None, matrix=None, weights=None, src=None, k=1
):
    # TODO: Do k-medoids algorithms assume triangle inequality? if so does it harm?
    # TODO: Weights are not applied in current implementation.
    assert (score_function is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, score_function, [src] * len(hyp))

    # Convert a similarit matrix into distance matrix and clip it into [0, 1] range.
    def dist_func(x, y):
        x_ = x.astype(int)
        y_ = y.astype(int)
        return max(min(1.0 - matrix[x_[0]][y_[0]], 1.0), 0.0)

    n = matrix.shape[0]
    X = np.arange(n).reshape((n, 1))  # Unflatten so that it can be used in sklearn.

    kmedoids = KMedoids(
        n_clusters=k, metric=dist_func, method="pam", init="k-medoids++"
    ).fit(X)

    k_bests = kmedoids.cluster_centers_.astype(int)[:, 0]
    # print('k_bests=', k_bests)
    return k_bests


if __name__ == "__main__":
    import os
    from parser import get_mbr_parser

    from tqdm import tqdm
    from utils import output_dir, score_dir

    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    n_lines = args.n_lines

    n_samples = args.n_samples
    eps = args.eps
    metric = args.metric

    # s_samples = args.s_samples
    strategy = args.algorithm
    k = args.k
    div_pen = args.div_pen

    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)

    for sample_id in tqdm(range(n_lines)):

        score_path = os.path.join(
            score_dir, dataset, "{:04d}_eps-{:.2f}_{}".format(sample_id, eps, metric)
        )

        score_matrix = np.loadtxt(score_path)  # .reshape((n_samples, n_samples))

        score_matrix = score_matrix[:n_samples, :n_samples]

        seq_scores = score_matrix.mean(axis=0)

        if strategy == "mbr":
            output = [np.argmax(seq_scores)]
            strategy_name = strategy
        elif strategy == "dmbr":
            output = compute_dmbr(score_matrix, k, div_pen=div_pen)
            strategy_name = "{}_{}_{:.2f}".format(strategy, k, div_pen)
        else:
            assert False

        output_path = os.path.join(
            output_dir, dataset, "{:04d}_{}".format(sample_id, strategy_name)
        )
        np.savetxt(output_path, np.array(output), fmt="%d")
