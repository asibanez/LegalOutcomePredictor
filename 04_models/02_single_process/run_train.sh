INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\01_article_split\\art_06_50p
OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\02_runs\\art_06\\batch_02\\38_TEST
PATH_EMBED=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\01_preprocessed\\id_2_embed_dict.pkl

#INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/01_article_split/art_06_50p
#OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/02_runs/art_06/34_art6_50p_att_TEST
#PATH_EMBED=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/01_preprocessed/id_2_embed_dict.pkl

python -m pdb train.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --path_embed=$PATH_EMBED \
    --n_epochs=30 \
    --batch_size=20 \
    --lr=1e-4 \
    --wd=0.1e-5 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=512 \
    --num_passages=50 \
    --embed_dim=200 \
    --hidden_dim=100 \
    --att_dim=100\
    --pad_idx=0 \
    --save_final_model=True \
    --save_model_steps=False \
    --use_cuda=False \
    --gpu_ids=2,3,4,5,6,7

read -p 'EOF'


#    --n_epochs=30 \
#    --batch_size=1000 \
