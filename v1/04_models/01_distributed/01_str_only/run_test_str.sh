INPUT_DIR=s3://cortex-mit1004-lmdl-workbucket/dev-asibanez/preprocessed/multihop/preprocessed_v2/str_datasets
WORK_DIR=/mnt/LMB56FBA268C66/legal-nlp-research/00_MIT_model/05_multihop_model/01_runs/02_only_str/run_03_DISTRIB

python test_str.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --batch_size=20000 \
    --gpu_id=0
  