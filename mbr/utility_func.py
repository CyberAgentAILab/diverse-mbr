# TODO: Put all the utility functions here.
# import re
import numpy as np
import torch
import torch.nn.functional as F
from comet import download_model, load_from_checkpoint
from distinct_n.utils import ngrams
from evaluate import load
from nltk.tokenize import ToktokTokenizer
from torch.nn.functional import cosine_similarity
from torchmetrics.text.infolm import InfoLM
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)


def load_similarity(sim):

    if sim == "bleurt":
        similarity = load(sim, checkpoint="BLEURT-20")

        def compute_similarity(hyp, ref, src):
            return similarity.compute(predictions=hyp, references=ref)["scores"]

    elif sim == "comet":
        similarity = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

        def compute_similarity(hyp, ref, src):
            data = []
            # print('src=', src)
            for i in range(len(hyp)):
                d = {}
                d["src"] = src[i]
                d["mt"] = hyp[i]
                d["ref"] = ref[i]
                data.append(d)
            model_output = similarity.predict(data, batch_size=64)
            return model_output.scores

    elif sim == "comet20":
        similarity = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))

        def compute_similarity(hyp, ref, src):
            data = []
            # print('src=', src)
            for i in range(len(hyp)):
                d = {}
                d["src"] = src[i]
                d["mt"] = hyp[i]
                d["ref"] = ref[i]
                data.append(d)
            model_output = similarity.predict(data, batch_size=128)
            return model_output.scores

    elif sim == "bertscore":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            return similarity.compute(predictions=hyp, references=ref, lang="en")["f1"]

    elif sim == "deberta":
        # This is a better bertscore model. Not sure how much it helps.
        similarity = load("bertscore")

        def compute_similarity(hyp, ref, src):
            return similarity.compute(
                predictions=hyp,
                references=ref,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
            )["f1"]

    elif sim == "sacrebleu":
        similarity = load(sim)

        def compute_similarity(hyp, ref, src):
            scores = [
                similarity.compute(predictions=[hyp[i]], references=[ref[i]])["score"]
                for i in range(len(hyp))
            ]
            return scores

    elif sim == "infolm":
        similarity = InfoLM(
            "google/bert_uncased_L-2_H-128_A-2",
            information_measure="fisher_rao_distance",
            idf=False,
            return_sentence_level_score=True,
        )

        def compute_similarity(hyp, ref, src):
            return -np.array(similarity(hyp, ref)[1])

    elif sim == "clip":
        # This computes the RefCLIPScore, not the reference-less CLIPScore.
        # TODO: there is no similarity function for this
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        # model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)

        model = CLIPModel.from_pretrained(model_id).to(device)
        similarity = CLIPTextModel.from_pretrained(model_id).to(device)
        model.eval()
        similarity.eval()

        def compute_similarity(hyp, ref, src):
            with torch.no_grad():
                hyp = list(hyp)
                ref = list(ref)
                inputs = processor(
                    text=hyp + ref,
                    images=src[0],
                    return_tensors="pt",
                    padding="max_length",
                ).to("cuda")

                text_embeddings = torch.flatten(
                    similarity(inputs.input_ids.to(device))["last_hidden_state"], 1, -1
                )
                hyp_embeddings = text_embeddings[: len(hyp)]
                ref_embeddings = text_embeddings[len(hyp) :]
                text_scores = (
                    cosine_similarity(hyp_embeddings, ref_embeddings)
                    .cpu()
                    .detach()
                    .numpy()
                )
                # print('text_scores.shape=', text_scores.shape)

                # Assume the src is the same for all the hypotheses.
                # TODO: Reuse the embedding
                img_inputs = processor(
                    text=hyp, images=src[0], return_tensors="pt", padding="max_length"
                ).to("cuda")
                img_outputs = model(**img_inputs)

                img_scores = np.squeeze(
                    (img_outputs.logits_per_image / 100).cpu().detach().numpy()
                )
                # print('img_scores.shape=', img_scores.shape)

                harmonic_mean = (
                    2 * text_scores * img_scores / (text_scores + img_scores)
                )
            # print('harmonic_mean=', harmonic_mean)
            return harmonic_mean

    elif sim == "cliptext":
        # This computes the RefCLIPScore, not the reference-less CLIPScore.
        # TODO: there is no similarity function for this
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        # model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)

        similarity = CLIPTextModel.from_pretrained(model_id).to(device)
        similarity.eval()

        def compute_similarity(hyp, ref, src):
            with torch.no_grad():
                hyp = list(hyp)
                ref = list(ref)
                inputs = processor(
                    text=hyp + ref,
                    images=src[0],
                    return_tensors="pt",
                    padding="max_length",
                ).to("cuda")

                text_embeddings = torch.flatten(
                    similarity(inputs.input_ids.to(device))["last_hidden_state"], 1, -1
                )
                hyp_embeddings = text_embeddings[: len(hyp)]
                ref_embeddings = text_embeddings[len(hyp) :]
                text_scores = (
                    cosine_similarity(hyp_embeddings, ref_embeddings)
                    .cpu()
                    .detach()
                    .numpy()
                )

            return text_scores

    elif sim == "unigramf1":
        similarity = ToktokTokenizer()

        def compute_similarity(hyp, ref, src):
            nhyp = len(hyp)
            f1s = []
            for i in range(nhyp):
                h = hyp[i]
                r = ref[i]
                hyp_tok = similarity.tokenize(h)
                ref_tok = similarity.tokenize(r)

                if len(hyp_tok) == 0 or len(ref_tok) == 0:
                    f1s.append(0.0)
                else:
                    precision = len(
                        [token for token in hyp_tok if token in ref_tok]
                    ) / len(hyp_tok)
                    recall = len(
                        [token for token in hyp_tok if token in ref_tok]
                    ) / len(ref_tok)

                    if precision + recall < 0.0001:
                        # Prevent zero division.
                        f1s.append(0.0)
                    else:
                        f1s.append(2.0 * precision * recall / (precision + recall))
            return f1s

    elif sim == "sentbert":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        evaluator = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        evaluator.eval()
        similarity = None

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def compute_similarity(hyp, ref, src):
            hyp = list(hyp)
            ref = list(ref)
            # print('hyp=', hyp)
            # print('ref=', ref)
            with torch.no_grad():
                encoded_input = tokenizer(
                    hyp + ref, padding=True, truncation=True, return_tensors="pt"
                )
                model_output = evaluator(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
                # print("sentence_embeddings_norm=", sentence_embeddings_norm)
                text_scores = []
                for i in range(len(hyp)):
                    text_score = (
                        cosine_similarity(
                            sentence_embeddings_norm[i : i + 1],
                            sentence_embeddings_norm[len(hyp) + i : len(hyp) + i + 1],
                        )
                        .cpu()
                        .detach()
                        .numpy()
                        .max()
                    )
                    text_scores.append(text_score)
            return text_scores

    else:
        assert False

    return compute_similarity, similarity


def load_distance(sim, compute_similarity):
    if sim != "sacrebleu":

        def compute_distance(hyp, ref, src):
            return [1.0 - sim for sim in compute_similarity(hyp, ref, src)]

    else:
        # sacrebleu ranges (0, 100), so need to normalize it.
        def compute_distance(hyp, ref, src):
            return [1.0 - sim / 100.0 for sim in compute_similarity(hyp, ref, src)]

    return compute_distance


def load_evaluate(eval_func, sim, similarity):

    if eval_func == "bleurt":
        evaluator = load(eval_func, checkpoint="BLEURT-20")
    elif eval_func == "comet":
        evaluator = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    elif eval_func == "comet20":
        evaluator = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
    elif eval_func == "clip":
        pass
    elif eval_func == "infolm":
        evaluator = InfoLM(
            "google/bert_uncased_L-2_H-128_A-2",
            information_measure="fisher_rao_distance",
            idf=False,
        )
    elif eval_func == "sentbert":
        pass
    else:
        evaluator = load(eval_func)

    if eval_func == "rouge":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]

    elif eval_func == "sacrebleu":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["score"]

    elif eval_func == "sacrebleuzh":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(
                predictions=[hyp], references=[ref], tokenize="zh"
            )["score"]

    elif eval_func == "bleurt":

        def compute_evaluate(hyp, ref, src):
            return evaluator.compute(predictions=[hyp], references=[ref])["scores"][0]

    elif eval_func == "comet":

        def compute_evaluate(hyp, ref, src):
            d = {"src": src, "mt": hyp, "ref": ref}
            data = [d]
            model_output = evaluator.predict(data, progress_bar=False)
            return model_output.scores[0]

    elif eval_func == "comet20":

        def compute_evaluate(hyp, ref, src):
            d = {"src": src, "mt": hyp, "ref": ref}
            data = [d]
            model_output = evaluator.predict(data, progress_bar=False)
            return model_output.scores[0]

    elif eval_func == "infolm":

        def compute_evaluate(hyp, ref, src):
            return np.array(evaluator(hyp, ref)).item()

    elif eval_func == "meteor":

        def compute_evaluate(hyp, ref, src):
            scores = [
                evaluator.compute(predictions=[hyp], references=[r])["meteor"]
                for r in ref
            ]
            return max(scores)

    elif eval_func == "clip":
        # This computes the RefCLIPScore, not the reference-less CLIPScore.
        # TODO: there is no similarity function for this
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "openai/clip-vit-large-patch14"
        # model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)

        model = CLIPModel.from_pretrained(model_id).to(device)
        evaluator = CLIPTextModel.from_pretrained(model_id).to(device)
        model.eval()
        evaluator.eval()

        def compute_evaluate(hyp, ref, src):
            with torch.no_grad():
                inputs = processor(
                    text=[hyp] + ref,
                    images=src,
                    return_tensors="pt",
                    padding="max_length",
                ).to("cuda")

                text_embeddings = torch.flatten(
                    evaluator(inputs.input_ids.to(device))["last_hidden_state"], 1, -1
                )
                hyp_embeddings = text_embeddings[:1]
                ref_embeddings = text_embeddings[1:]
                text_scores = (
                    cosine_similarity(hyp_embeddings, ref_embeddings)
                    .cpu()
                    .detach()
                    .numpy()
                    .max()
                )
                # print('text_scores.shape=', text_scores.shape)

                # Assume the src is the same for all the hypotheses.
                # TODO: Reuse the embedding
                img_inputs = processor(
                    text=hyp, images=src, return_tensors="pt", padding="max_length"
                ).to("cuda")
                img_outputs = model(**img_inputs)

                img_scores = np.squeeze(
                    (img_outputs.logits_per_image / 100).cpu().detach().numpy()
                )
                # print('img_scores.shape=', img_scores.shape)

                harmonic_mean = (
                    2 * text_scores * img_scores / (text_scores + img_scores)
                )
            # print('harmonic_mean=', harmonic_mean)
            return harmonic_mean

    elif eval_func == "sentbert":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        evaluator = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        evaluator.eval()

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        def compute_evaluate(hyp, ref, src):
            with torch.no_grad():
                encoded_input = tokenizer(
                    [hyp, ref], padding=True, truncation=True, return_tensors="pt"
                )
                model_output = evaluator(**encoded_input)

                # Perform pooling
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input["attention_mask"]
                )
                sentence_embeddings_norm = F.normalize(sentence_embeddings, p=2, dim=1)
                # print("sentence_embeddings_norm=", sentence_embeddings_norm)
                text_scores = (
                    cosine_similarity(
                        sentence_embeddings_norm[:1], sentence_embeddings_norm[1:]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                    .max()
                )
            return text_scores

    else:
        assert False

    return compute_evaluate, evaluator


def compute_self_score(hyps, src, compute_evaluate):
    scores = []
    n_samples = 0
    n = len(hyps)
    for i in range(n):
        for j in range(n):
            if i != j:
                score = compute_evaluate(hyps[i], hyps[j], src)
                scores.append(score)
                n_samples += 1
    return sum(scores) / n_samples


def distinct_n_diversity(sentences, n):
    """
    Compute distinct-N among a set of sentences.
    :param sentences: a list of sentences.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    assert n >= 1
    assert isinstance(sentences, list)
    if len(sentences) == 0:
        return 0.0  # Prevent a zero division
    assert isinstance(sentences[0], str)

    word_tokenizer = ToktokTokenizer()

    list_of_words = [word_tokenizer.tokenize(sentence) for sentence in sentences]

    distinct_ngrams = set()
    for words in list_of_words:
        # if len(words) == 0:
        #     continue
        if len(words) < n:
            continue
        d_ngrams = ngrams(words, n)
        distinct_ngrams.update(d_ngrams)

    return len(distinct_ngrams) / sum([len(words) for words in list_of_words])


def evaluate_diversity(hyp, scores, src_input, compute_pairwise):
    """
    This function computes the metrics for the diversity experiments.
    kmbr_mean_score: mean score of the hypotheses.
    kmbr_min_score: min score of the hypotheses.
        -> These two metrics are used to compare the quality of the hypotheses.
    """
    # print('hyp=', hyp)
    kmbr_mean_score = sum(scores) / len(scores)
    kmbr_min_score = min(scores)
    kmbr_max_score = max(scores)
    kmbr_self_score = compute_self_score(hyp, src_input, compute_pairwise)
    kmbr_dn_1 = distinct_n_diversity(hyp, 1)
    kmbr_dn_2 = distinct_n_diversity(hyp, 2)
    kmbr_dn_3 = distinct_n_diversity(hyp, 3)
    return [
        kmbr_mean_score,
        kmbr_min_score,
        kmbr_max_score,
        kmbr_self_score,
        kmbr_dn_1,
        kmbr_dn_2,
        kmbr_dn_3,
    ]
