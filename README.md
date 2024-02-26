# Diverse Minimum Bayes Risk Decoding

This repository contains the code for the experiments in [Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding](https://arxiv.org/abs/2401.05054).

The code is provided mostly as is with little effort on refactoring.

## Installation

```
git clone git@github.com/jinnaiyuu/diverse-mbr
cd diverse-mbr
pip install requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh ` computes the MBR candidate from the candidates sampled.

### Sampling candidates

```
./experiments/sample.sh -d [DATASET] -s [NUMBER OF SAMPLES] 
```

### Computing Diverse MBR and KMBR

```
./experiments/run_mbr.sh -d [DATASET] -s [NUMBER OF SAMPLES] -a [ALGORITHM]
```

### Example on WMT'19 En-De

1. Use [sacrebleu](https://github.com/mjpost/sacrebleu) to prepare the benchmark dataset.
```
sacrebleu -t wmt19 -l en-de --echo src > ./dataset/wmt19-text/wmt19.en-de.en
sacrebleu -t wmt19 -l en-de --echo ref > ./dataset/wmt19-text/wmt19.en-de.de
```

2. Sample candidates on WMT'19 En-De

```
./experiments/sample.sh -d wmt19.en-de
```

3. Computing the Diverse MBR output on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de -a diverse
```

4. Computing the k-Medoid MBR output on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de -a kmmbr
```


## Reference

[Jinnai, Y., Honda, U., Morimura, T., & Zhang, P. (2024). Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding. arXiv preprint arXiv:2401.05054.](https://arxiv.org/abs/2401.05054)

Bibtex:
```
@article{jinnai2024generating,
      title={Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding}, 
      author={Yuu Jinnai and Ukyo Honda and Tetsuro Morimura and Peinan Zhang},
      year={2024},
      journal={arXiv preprint arXiv:2401.05054}
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.
