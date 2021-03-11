#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\01_article_split\\art_06_50p_par_SHORT
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\02_runs\\batch_02\\art_06\\38_TEST
#PATH_EMBED=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\id_2_embed_dict.pkl
#PATH_MODEL=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_repo\\04_models\\02_single_process\\model_attention_v4\\model_attn_v4.py

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_06_50p_par
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/02_runs/02_batch_02/art_06/12_TEST_DELETE
PATH_EMBED=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/id_2_embed_dict.pkl
PATH_MODEL=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/01_repo/04_models/02_single_process/model_attention_v4/model_attn_v4.py

python train.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --path_embed=$PATH_EMBED \
    --path_model=$PATH_MODEL \
    --n_epochs=2\
    --batch_size=500 \
    --lr=1e-4 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=512 \
    --num_passages=50 \
    --num_par_arts=11 \
    --embed_dim=200 \
    --hidden_dim=100 \
    --att_dim=100 \
    --pad_idx=0 \
    --save_final_model=True \
    --save_model_steps=False \
    --use_cuda=True \
    --gpu_ids=5,6,7

#read -p 'EOF'

#--batch_size=150
#--n_epochs=30
