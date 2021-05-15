#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full
WORK_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/17_BERT_TRANSF_v2_FIX_50par_10ep_rationale_v1_TEST_DELETE

python test_v2.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --test_file=model_dev.pkl \
    --batch_size=1000 \
    --seq_len=256 \
    --num_labels=33 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=50 \
    --pad_idx=0 \
    --gumbel_temp=1 \
    --T_s=0.3 \
    --lambda_s=0.1 \
    --gpu_id=0

#read -p 'EOF'

