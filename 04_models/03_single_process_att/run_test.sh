#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_data\\01_preprocessed\\01_article_split\\art_06_50p
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\05_runs\\art_06\\38_TEST
#PATH_EMBED=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_data\\01_preprocessed\\id_2_embed_dict.pkl
#PATH_MODEL=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\01_repo\\04_models\\02_single_process\\model_attention_v4\\model_attn_v4.py

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_06_50p_par_att
WORK_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/02_runs/02_batch_02/art_06/12_art_3_5_6_13_50p_att_v4_2_50ep_att
PATH_EMBED=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/id_2_embed_dict.pkl
PATH_MODEL=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/01_repo/04_models/03_single_process_att/model_attention_v4/model_attn_v4_3.py

python -m pdb test.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --path_embed=$PATH_EMBED \
    --path_model=$PATH_MODEL \
    --batch_size=300 \
    --seq_len=512 \
    --num_passages=50 \
    --num_par_arts=11 \
    --embed_dim=200 \
    --hidden_dim=100 \
    --att_dim=100 \
    --pad_idx=0 \
    --gpu_id=0

#read -p 'EOF'
