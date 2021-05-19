#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full
WORK_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/24_BERT_TRANSF_v2_FIX_50par_30ep_rationale_v2_mod_v6_rationale_null_lambda_temp_10_no_drops

python test_v2.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --test_file=model_train.pkl \
    --batch_size=1000 \
    --seq_len=256 \
    --num_labels=33 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=50 \
    --pad_idx=0 \
    --rationales=True \
    --gumbel_temp=10 \
    --T_s=0.3 \
    --lambda_s=0 \
    --gpu_id=0

#read -p 'EOF'

