#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/00_full
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/00_TEST_DELETE_2

python train.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --n_epochs=10 \
    --batch_size=2000 \
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
    --save_model_steps=False \
    --use_cuda=True \
    --gpu_ids=2,3,4,5

#read -p 'EOF'

#--batch_size=150
#--n_epochs=30
#--max_n_pars=200
