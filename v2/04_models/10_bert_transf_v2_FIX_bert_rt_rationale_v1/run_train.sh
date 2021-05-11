#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/01_full
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/16_BERT_TRANSF_v2_FIX_50par_20ep_rationale_v1

python -m ipdb train_###.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --n_epochs=2 \
    --batch_size=4 \
    --shuffle_train=True \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=256 \
    --num_labels=33 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=50 \
    --pad_idx=0 \
    --T_s=0.3 \
    --lambda_s=0.1 \
    --save_final_model=True \
    --save_model_steps=True \
    --use_cuda=True \
    --gpu_ids=0

#read -p 'EOF'

#--batch_size=40
#--n_epochs=20
#--max_n_pars=200
