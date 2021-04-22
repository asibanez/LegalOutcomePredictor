#INPUT_DIR=s3://cortex-mit1004-lmdl-workbucket/dev-asibanez/preprocessed/multihop/preprocessed_v2/toy_datasets
INPUT_DIR=s3://cortex-mit1004-lmdl-workbucket/dev-asibanez/preprocessed/multihop/preprocessed_v2/full_datasets

WORK_DIR=/mnt/LMB56FBA268C66/legal-nlp-research/00_MIT_model/05_multihop_model/01_runs/01_only_text/run_01_test_DISTRIB
AWS_BUCKET_NAME=cortex-mit1004-lmdl-workbucket
PATH_EMB_LEAF_NODES=data/models/TrainFastText/train_fast_text__combine_all_policy_document_text_8af01292.pkl
PATH_EMB_LOSS_DESC=data/models/TrainFastText/train_fast_text__download_loss_description_4c6a8dde.pkl

python test_text.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --aws_bucket_name=$AWS_BUCKET_NAME \
    --path_emb_leaf_nodes=$PATH_EMB_LEAF_NODES \
    --path_emb_loss_desc=$PATH_EMB_LOSS_DESC \
    --batch_size=300 \
    --seq_len=512 \
    --num_leaf_nodes=35 \
    --embed_dim=300 \
    --hidden_dim=100 \
    --att_dim=100 \
    --pad_idx=0 \
    --gpu_id=0
