# SERVER: Aux script for preprocessing in server
# Input:  List with tokenized corpus including '<UNK>' tokens in pkl format
# Output: String with corpus including '<UNK>' tokens in txt format

import sys
import codecs
import pickle

def main(input_path, output_path):
    
    with open(input_path, 'rb') as fr:
        corpus_tok = pickle.load(fr)

    corpus_txt = (' ').join(corpus_tok)

    with codecs.open(output_path, 'w', 'utf-8') as fw:
        fw.write(corpus_txt)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: 08_gen_corpus_txt.py [input path] [output path]')
        exit()
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)
