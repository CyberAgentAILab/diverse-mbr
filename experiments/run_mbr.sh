# Default parameters are set to run a debug experiment.

DOMAIN=wmt19.en-de
MODEL=wmt19-en-de
NLINES=3
NSAMPLES=5
EPS=0.01
TOPK=0
TOPP=1.0
SIM=bertscore
EVAL=sacrebleu
ALGORITHM=None
DEBUG=0
RECOMPUTE=""

DOSAMPLE=1
DIVERSITY=1.0
DIVERSEK=4
PAIRWISE=sacrebleu

BUDGETS=-1

while getopts d:m:p:l:s:e:k:n:i:v:a:bru:t:z:w:o:h: option
do
  case $option in
    d)
        DOMAIN=${OPTARG};;
    m)
        MODEL=${OPTARG};;
    p)
        PROMPT=${OPTARG};;
    l)
        NLINES=${OPTARG};;
    s)
        NSAMPLES=${OPTARG};;
    e)
        EPS=${OPTARG};;
    k)
        TOPK=${OPTARG};;
    n)
        # Nucleus sampling
        TOPP=${OPTARG};;
    i)
        # TODO: Long options
        SIM=${OPTARG};;
    v)
        EVAL=${OPTARG};;
    a)
        ALGORITHM=${OPTARG};;
    b)
        DEBUG=1;;
    r)
        RECOMPUTE="--recompute";;
    t)
        # DIVERSITY
        DOSAMPLE=0
        DIVERSITY=${OPTARG};;
    z)
        # DIVERSITY
        DIVERSEK=${OPTARG};;
    w)
        # DIVERSITY
        PAIRWISE=${OPTARG};;
    \?)
      echo "This is unexpected option." 1>&2
      exit 1
  esac
done

set -e

python3 mbr/mbr_engine.py $DOMAIN \
    --model $MODEL \
    --sample_dir ./samples/$DOMAIN/$MODEL \
    --n_lines $NLINES --n_samples $NSAMPLES \
    --eps $EPS --topk $TOPK --topp $TOPP \
    --sim $SIM \
    --eval $EVAL \
    --algorithm $ALGORITHM \
    $RECOMPUTE \
    --do_sample $DOSAMPLE --diversity_penalty $DIVERSITY \
    --diverse_k $DIVERSEK \
    --pairwise_eval $PAIRWISE

if [ "$DEBUG" == "1" ]; then
    echo "done!"

fi
