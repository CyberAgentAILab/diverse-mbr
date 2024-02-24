import numpy as np

def compute_score_matrix(samples, score_function, src_input=None):
    # TODO: add param ref_samples to compute the score between two different samples.
    n_samples = len(samples)
    scores = []
    for i in range(n_samples):
        score = score_function(hyp=np.array([samples[i]] * n_samples), ref=samples, src=src_input)
        scores.append(score)
    return np.array(scores)

def compute_mbr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, incremental=False):
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))
        
    if weights is not None:
        mbr_scores = matrix @ np.transpose(weights)
    else:
        mbr_scores = np.sum(matrix, axis=1)
        
    if incremental:
        best_hyp = -1
        best_score = -np.inf
        bests = []
        for i in range(mbr_scores.shape[0]):
            if mbr_scores[i] > best_score:
                best_hyp = i
                best_score = mbr_scores[i]
            assert best_hyp >= 0
            bests.append(best_hyp)
        return bests # List of hypothesis indices.
    else:
        best_hyp = np.argmax(mbr_scores)
        
        assert best_hyp >= 0
        return best_hyp

def compute_kmbr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, k=1):
    # Used for 1. Coarse to fine and 2. Baseline for kMBR algorithms.
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))
        
    if weights is not None:
        mbr_scores = matrix @ np.transpose(weights)
    else:
        mbr_scores = np.sum(matrix, axis=1)
    
    # The hypotheses don't have to be sorted but it may be useful somewhere.
    best_hyps = np.argsort(-mbr_scores)[:k]
    
    return best_hyps

def compute_nbys_mbr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, T_budget=None, randomized=False):
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))
    
    n_samples = matrix.shape[0]
    
    s = T_budget // n_samples + 1

    # s = n_samples
    # s_budget = s * (s - 1) / 2 + s * (n_samples - s)
    # while s_budget > T_budget:
    #     s -= 1
    #     s_budget = s * (s - 1) / 2 + s * (n_samples - s)
    if not randomized:
        return compute_mbr(matrix=matrix[:, :s])
    else:
        pick = np.random.choice(np.arange(matrix.shape[1]), size=s, replace=False)
        picked_matrix = np.take(matrix, pick, axis=1)
        return compute_mbr(matrix=picked_matrix)

def compute_c2f_mbr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, T_budget=None, 
                    compute_coarse=None, coarse_matrix=None):
    assert hyp is not None
    assert (compute_coarse is not None) or (coarse_matrix is not None)

    n_samples = matrix.shape[0]

    nk = T_budget // n_samples + 1
    
    # # Compute the number of samples to use for coarse to fine.
    # nk = n_samples
    # s_budget = nk * (nk - 1) / 2 + nk * (n_samples - nk)
    # while s_budget > T_budget:
    #     nk -= 1
    #     s_budget = nk * (nk - 1) / 2 + nk * (n_samples - nk)
        
    # Compute coarse measure and pick the best k as a candidate.
    if coarse_matrix is None:
        coarse_matrix = compute_score_matrix(hyp, compute_coarse, [src] * len(hyp))
    coarse_bests = compute_kmbr(matrix=coarse_matrix, k=nk)
        
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))
    
    # Pick the best from the coarse set.
    # The candidates are limited but the references are set the same.
    compressed_matrix = matrix[coarse_bests]
    
    # TODO: Here the compressed matrix is NOT sorted in a way that diagonal elements
    # are comparing to the same hypothesis. It doesn't matter in the current implementation,
    # but in future it may be so better refactor it. 
    best_in_comp_ind = compute_mbr(matrix=compressed_matrix)
    
    best_in_orig_ind = coarse_bests[best_in_comp_ind]
    
    return best_in_orig_ind