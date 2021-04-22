INPUT_DIR=s3://cortex-mit1004-lmdl-workbucket/dev-asibanez/preprocessed/multihop/preprocessed_v2/str_datasets
WORK_DIR=/mnt/LMB56FBA268C66/legal-nlp-research/00_MIT_model/05_multihop_model/01_runs/02_only_str/run_03_DISTRIB

python train_val_str.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$WORK_DIR \
    --n_epochs=25 \
    --batch_size=20000 \
    --lr=0.01 \
    --wd=0.00001 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --save_final_model=True \
    --save_model_steps=False \
    --nodes=1 \
    --gpus=6 \
    --nr=0
