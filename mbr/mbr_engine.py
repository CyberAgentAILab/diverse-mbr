import argparse
import json
import os
from parser import get_mbr_parser

import boto3
import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint
from evaluate import load
from policy.diverse_mbr import compute_dmbr, compute_kmedmbr
from policy.mbr import (
    compute_c2f_mbr,
    compute_kmbr,
    compute_mbr,
    compute_nbys_mbr,
    compute_score_matrix,
)
from tqdm import tqdm
from utility_func import *
from utils import load_matrix  # , approx_dir, diverse_dir
from utils import load_dataset, load_samples_from_file, matrix_dir, result_dir


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir

    n_lines = args.n_lines
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix

    diverse_k = args.diverse_k
    diversity_penalty = args.diversity_penalty
    pairwise_eval = args.pairwise_eval

    do_sample = args.do_sample > 0

    if algorithm == "dbs":
        assert not do_sample
    else:
        assert do_sample

    diverse_pens = [0.1, 0.3, 0.5, 0.8, 1.0, 2.0]

    # TODO: similarity function is not required when the matrix is already computed.
    # Load utility function and evaluation functions
    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)

    if algorithm in ["dbs", "diverse", "diversesample"]:
        compute_pairwise, _ = load_evaluate(pairwise_eval, sim, similarity)

    # Load dataset
    # It might be easier to load src whatsoever
    # if ('comet' in sim) or ('clip' in sim):
    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)
    # os.makedirs(os.path.join(approx_dir, dataset, model_n), exist_ok=True)
    # os.makedirs(os.path.join(diverse_dir, dataset, model_n), exist_ok=True)
    # sample_dir = os.path.join('samples', dataset, model_n)

    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(
        files, epsilon, topk, topp, do_sample, diverse_k, diversity_penalty
    )

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rows = []

    if algorithm == "c2f":
        compute_coarse_sim, _ = load_similarity("sacrebleu")
    elif algorithm == "c2ff1":
        compute_coarse_sim, _ = load_similarity("unigramf1")

    for filename in filtered_files:

        sample_id = int(filename.split("_")[0])
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break

        # if ('comet' in sim) or ('clip' in sim):
        #     src_input = src_lines[sample_id]
        # else:
        #     src_input = None
        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]
        # print('src=', src)
        # print('trg=', trg)

        try:
            df = pd.read_csv(os.path.join(sample_dir, filename))
        except:
            print(
                os.path.join(sample_dir, filename),
                "is not readable with default engine.",
            )
            continue
            # df = pd.read_csv(os.path.join(sample_dir, filename), engine='python')

        if algorithm != "dbs":
            assert len(df) >= n_samples
            df = df[:n_samples]
        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp = df.iloc[:]["text"]

        # TODO: Should factor out dbs as it is very different experiment.
        if algorithm != "dbs":
            if not recompute_matrix:
                # This makes loading a matrix of size larger
                matrix = load_matrix(
                    os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
                )
                # if not (matrix is None):
                #     print('matrix loaded and reshaped to', matrix.shape)
            else:
                matrix = None
            if matrix is None:
                matrix_filename = filename + "_" + sim + "_" + str(n_samples)
                matrix_path = os.path.join(
                    matrix_dir, dataset, model_n, matrix_filename
                )

                # TODO: It may not needed when running approx, but for now let's assume we need a matrix.
                #       Need to refactor it later on.

                # if not (algorithm == "approx"):
                # Approx may not need the matrix to begin with.
                matrix = compute_score_matrix(
                    hyp, compute_similarity, [src_input] * len(hyp)
                )
                np.savetxt(matrix_path, matrix)

                # else:
                #     matrix = None

        if algorithm == "incremental":
            # TODO: We should merge this to the exact algorithm. For simplicity we keep it separate for now.
            # MBR: Monte Carlo Estimate
            ed_bests = compute_mbr(matrix=matrix, incremental=True)
            cache = {}
            ed_scores = []
            for ed_best in ed_bests:
                if ed_best in cache.keys():
                    ed_score = cache[ed_best]
                else:
                    ed_score = compute_score(
                        df, ed_best, trg, compute_evaluate, src=src_input
                    )
                    cache[ed_best] = ed_score
                ed_scores.append(ed_score)
            # ed_scores = [compute_score(df, ed_best, trg, compute_evaluate, src=src_input) for ed_best in ed_bests]
            row = [[sample_id, ed_scores[i], ed_bests[i]] for i in range(len(ed_bests))]

        elif algorithm == "diverse":
            # TODO: Add the coverage for CommonGen dataset. need lemmatization, lower. nltk.download()

            # Naive MBR (just select the top-k without considering diversity)
            kmbr_bests = compute_kmbr(matrix=matrix, k=diverse_k)
            # print('kmbr_bests=', kmbr_bests)
            kmbr_hyps = df["text"].iloc[kmbr_bests].to_list()
            kmbr_scores = [
                compute_score(df, kmbr_bests[i], trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            kmbr_stats = evaluate_diversity(
                kmbr_hyps, kmbr_scores, src_input, compute_pairwise
            )

            # K-medoid MBR
            kmmbr_bests = compute_kmedmbr(matrix=matrix, k=diverse_k)
            kmmbr_hyps = df["text"].iloc[kmmbr_bests].to_list()
            kmmbr_scores = [
                compute_score(df, kmmbr_bests[i], trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            kmmbr_stats = evaluate_diversity(
                kmmbr_hyps, kmmbr_scores, src_input, compute_pairwise
            )

            row = (
                [sample_id, kmbr_scores, kmbr_bests]
                + kmbr_stats
                + [kmmbr_scores, kmmbr_bests]
                + kmmbr_stats
            )

            # Diverse MBR
            for diverse_pen in diverse_pens:
                dmbr_bests = compute_dmbr(
                    matrix=matrix, k=diverse_k, div_pen=diverse_pen
                )
                dmbr_hyps = df["text"].iloc[dmbr_bests].to_list()
                dmbr_scores = [
                    compute_score(
                        df, dmbr_bests[i], trg, compute_evaluate, src=src_input
                    )
                    for i in range(diverse_k)
                ]
                dmbr_stats = evaluate_diversity(
                    dmbr_hyps, dmbr_scores, src_input, compute_pairwise
                )
                row += [dmbr_scores, dmbr_bests] + dmbr_stats
        elif algorithm == "diversesample":
            dbs_hyps = df["text"].to_list()[:diverse_k]
            dbs_scores = [
                compute_score(df, i, trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            dbs_stats = evaluate_diversity(
                dbs_hyps, dbs_scores, src_input, compute_pairwise
            )
            row = [sample_id, dbs_scores, []] + dbs_stats
        elif algorithm == "dbs":
            dbs_hyps = df["text"].to_list()
            dbs_scores = [
                compute_score(df, i, trg, compute_evaluate, src=src_input)
                for i in range(diverse_k)
            ]
            dbs_stats = evaluate_diversity(
                dbs_hyps, dbs_scores, src_input, compute_pairwise
            )
            row = [sample_id, dbs_scores, []] + dbs_stats
        else:
            assert False
        rows.append(row)

    if algorithm == "incremental":
        # TODO: Add other algorithms if needed.
        columns = ["sample_id", "ed_score", "ed_best"]
        postfix = ""
    elif algorithm == "diverse":
        columns = ["sample_id"]
        methods = ["kmbr", "kmmbr"] + [
            "dmbr-{}".format(div_pen) for div_pen in diverse_pens
        ]
        metrics = [
            "_score",
            "_best",
            "_mean_score",
            "_min_score",
            "_max_score",
            "_self_score",
            "_dn_1",
            "_dn_2",
            "_dn_3",
        ]
        for method in methods:
            cl = [method + metric for metric in metrics]
            columns += cl

        postfix = "_diverse_{:02d}".format(diverse_k)

        if pairwise_eval != "sacrebleu":
            postfix += "_{}".format(pairwise_eval)
    elif algorithm == "diversesample":
        columns = ["sample_id"]
        methods = ["diversesample-{}".format(diversity_penalty)]
        metrics = [
            "_score",
            "_best",
            "_mean_score",
            "_min_score",
            "_max_score",
            "_self_score",
            "_dn_1",
            "_dn_2",
            "_dn_3",
        ]
        for method in methods:
            cl = [method + metric for metric in metrics]
            columns += cl
        postfix = "_diversesample_{:02d}".format(diverse_k)

        if pairwise_eval != "sacrebleu":
            postfix += "_{}".format(pairwise_eval)
    elif algorithm == "dbs":
        columns = ["sample_id"]
        methods = ["dbs-{}".format(diversity_penalty)]
        metrics = [
            "_score",
            "_best",
            "_mean_score",
            "_min_score",
            "_max_score",
            "_self_score",
            "_dn_1",
            "_dn_2",
            "_dn_3",
        ]
        for method in methods:
            cl = [method + metric for metric in metrics]
            columns += cl
        postfix = "_dbs_{:02d}_{:.2f}".format(diverse_k, diversity_penalty)
        if pairwise_eval != "sacrebleu":
            postfix += "_{}".format(pairwise_eval)
    else:
        assert False

    if algorithm != "incremental":
        df = pd.DataFrame(rows, columns=columns)

        if algorithm == "dbs":
            filename = "{}_{}_{}{}.csv".format(dataset, model_n, eval_func, postfix)
        else:
            filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(
                dataset,
                model_n,
                n_samples,
                epsilon,
                topk,
                topp,
                sim,
                eval_func,
                postfix,
            )

        df_path = os.path.join(result_dir, filename)
        df.to_csv(df_path, index=False)

    else:
        # This is for the incremental procedure.
        # print('len(rows)=', len(rows))
        for i, r in enumerate(rows):
            # print(i, 'len(row)=', len(r))
            assert len(r) == n_samples

        for i_n_samples in range(0, n_samples):
            i_rows = [r[i_n_samples] for r in rows]
            df = pd.DataFrame(i_rows, columns=columns)
            filename = "{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}.csv".format(
                dataset,
                model_n,
                i_n_samples + 1,
                epsilon,
                topk,
                topp,
                sim,
                eval_func,
                postfix,
            )

            df_path = os.path.join(result_dir, filename)
            df.to_csv(df_path, index=False)
