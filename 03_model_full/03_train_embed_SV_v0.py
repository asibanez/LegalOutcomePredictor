# SERVER: Script required for running fastext on server
# Input:  Corpus (including <UNK>') in txt format
# Output: Dictionary tok_2_id in pkl format
#         Dictionary id_2_embed in pkl format

import os
import sys
import tqdm
import pickle
import fasttext
import numpy as np

def main(input_path, output_folder, embedding_dim):

    # Path definition
    input_path = input_path
    output_path_tok2id = os.path.join(output_folder, 'tok_2_id_dict.pkl')
    output_path_id2embed = os.path.join(output_folder, 'id_2_embed_dict.pkl') 
    
    # Read corpus and train word vectors
    model = fasttext.train_unsupervised(input_path, dim = embedding_dim)
    
    # Generate id & embedding dictionaries
    # token to ID
    token_to_id = {}
    token_to_id['<PAD>'] = 0
    for idx, token in enumerate(tqdm.tqdm(sorted(model.words))):
        token_to_id[token] = idx + 1
    
    # ID to embeddings
    id_to_embedding = {}
    id_to_embedding[0] = np.float32(np.array([0] * embedding_dim))
    
    for token, id in tqdm.tqdm(token_to_id.items()):
        if id != 0:
            id_to_embedding[id] = model.get_word_vector(token)
    
    # Save dictionaries
    with open(output_path_tok2id, 'wb') as fw:
        pickle.dump(token_to_id, fw)
    
    with open(output_path_id2embed, 'wb') as fw:
        pickle.dump(id_to_embedding, fw)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: 03_train_embed_SV_v0.py [input path] [output folder] [embedding dim]')
        exit()
    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    embedding_dim = int(sys.argv[3])
    main(input_path, output_folder, embedding_dim)
