import os
import tqdm
import pickle
import fasttext
import numpy as np

#%% Path definition

base_folder = os.path.join(os.getcwd(),'01_data', '01_preprocessed')
input_path= os.path.join(base_folder, 'corpus.txt')
output_path_tok2id = os.path.join(base_folder, 'tok_2_id')
output_path_id2embed = os.path.join(base_folder, 'id_2_embed')

#%% Initialize variables

embedding_dim = 128

#%% Read corpus and train word vectors

model = fasttext.train_unsupervised(input_path, dim = embedding_dim)

#%% Generate id & embedding dictionaries

# token to ID
token_to_id = {}
token_to_id['<pad>'] = 0
for idx, token in enumerate(tqdm.tqdm(sorted(model.words))):
    token_to_id[token] = idx + 1

# ID to embeddings
id_to_embedding = {}
id_to_embedding[0] = np.array([0] * embedding_dim)

for token, id in tqdm.tqdm(token_to_id.items()):
    if id != 0:
        id_to_embedding[id] = model.get_word_vector(token)

#%% Save dictionaries

with open(output_path_tok2id, 'wb') as fw:
    pickle.dump(token_to_id, fw)

with open(output_path_id2embed, 'wb') as fw:
    pickle.dump(id_to_embedding, fw)
