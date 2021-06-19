#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/02_full_binary
WORK_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/01_binary/00_BERT_TRANSF_v2_FIX_50par_5ep

python -m ipdb test.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --test_file=model_test.pkl \
    --batch_size=1000 \
    --seq_len=256 \
    --num_labels=1 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars_facts=50 \
    --max_n_pars_echr=6 \
    --pad_idx=0 \
    --gpu_id=0

#read -p 'EOF'

