#INPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_toy_bert_encoded
#OUTPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/02_runs/06_TESTING_BERT_ENCODING_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/03_full_1_bert_encoded
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/09_TEST_BERT_TRANSF_v3_100ep_encod

python -m ipdb train.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --n_epochs=100 \
    --batch_size=800 \
    --shuffle_train=True \
    --lr=1e-4 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=512 \
    --num_labels=33 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=200 \
    --pad_idx=0 \
    --save_final_model=True \
    --save_model_steps=True \
    --use_cuda=True \
    --gpu_ids=0

#read -p 'EOF'

#--batch_size=4000
#--n_epochs=50
