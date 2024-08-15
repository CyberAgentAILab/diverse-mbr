# Diverse Minimum Bayes Risk Decoding

This repository contains the code for the experiments in [Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding](https://aclanthology.org/2024.findings-acl.503/).

The code is tested on Ubuntu 20.04 using Python 3.8 and CUDA 11.0 (Docker image nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04).
The code is provided mostly as is with little effort on refactoring.

## Installation

```
git clone git@github.com:CyberAgentAILab/diverse-mbr
cd diverse-mbr
pip install -r requirements.txt
```

## Usage

The code runs in two steps.
1. `sample.sh` samples candidates.
2. `run_mbr.sh` computes the MBR candidate from the candidates sampled.

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
mkdir -p ./dataset/wmt19-text
sacrebleu -t wmt19 -l en-de --echo src > ./dataset/wmt19-text/wmt19.en-de.en
sacrebleu -t wmt19 -l en-de --echo ref > ./dataset/wmt19-text/wmt19.en-de.de
```

2. Sample candidates on WMT'19 En-De

```
./experiments/sample.sh -d wmt19.en-de
```

3. Computing Diverse MBR and K-Medoid MBR on WMT'19 En-De

```
./experiments/run_mbr.sh -d wmt19.en-de -m wmt19-en-de -a diverse
```


## Reference

[Yuu Jinnai, Ukyo Honda, Tetsuro Morimura, and Peinan Zhang. 2024. Generating Diverse and High-Quality Texts by Minimum Bayes Risk Decoding. In Findings of the Association for Computational Linguistics ACL 2024, pages 8494–8525, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.](https://aclanthology.org/2024.findings-acl.503/)

Bibtex:
```
@inproceedings{jinnai-etal-2024-generating,
    title = "Generating Diverse and High-Quality Texts by Minimum {B}ayes Risk Decoding",
    author = "Jinnai, Yuu  and
      Honda, Ukyo  and
      Morimura, Tetsuro  and
      Zhang, Peinan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.503",
    pages = "8494--8525",
}
```

## Contact
For any questions, feel free to raise an issue or contact me at jinnai_yu@cyberagent.co.jp.


## Acknowledgements

[MS COCO dataset](https://cocodataset.org/#home) is licensed under a [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/).
