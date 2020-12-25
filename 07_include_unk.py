import os
import sys
import tqdm
import pickle

def main(start, stop):
    base_path = '/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/01_data/'
    output_file = 'corpus_' + str(start) + '_' + str(stop) + '.pkl'
    corpus_path = os.path.join(base_path, 'corpus_tok_list.pkl')
    vocab_path = os.path.join(base_path, 'vocab_list.pkl')
    output_path = os.path.join(base_path, output_file)

    with open(corpus_path, 'rb') as fr:
        corpus_tok = pickle.load(fr)

    with open(vocab_path, 'rb') as fr:
        vocab = pickle.load(fr)

    corpus_new = []
    
    for token in tqdm.tqdm(corpus_tok[start:stop]):
        if token in vocab:
            corpus_new.append(token)
        else:
            corpus_new.append('<UNK>')
    
    with open(output_path, 'wb') as fw:
        pickle.dump(corpus_new, fw)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: 07_unk.py [start] [stop]')
        exit()
    start = int(sys.argv[1])
    stop = int(sys.argv[2])
    main(start, stop)
