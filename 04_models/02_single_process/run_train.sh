#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\01_article_split\\art_06_50p_par_SHORT
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\02_runs\\batch_02\\art_06\\38_TEST
#PATH_EMBED=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\id_2_embed_dict.pkl

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_06_50p_par
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/02_runs/02_batch_02/art_06/10_art6_50p_att_v4_50ep
PATH_EMBED=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/id_2_embed_dict.pkl

python train.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --path_embed=$PATH_EMBED \
    --n_epochs=50 \
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
    --gpu_ids=0,1,3

#read -p 'EOF'

#--batch_size=150
#--n_epochs=30
