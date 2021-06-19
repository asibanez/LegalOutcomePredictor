# Input:  ECHR law in xml format
# Output: Dictionary {Article number: Article text} in pkl format 
# v1 -> Code cleaned

#%% Imports

import os
import pickle
from lxml import etree

#%% Define paths

base_folder = os.getcwd()
input_path = os.path.join(base_folder, '01_data', '00_original', 'ECHR.html' )
output_path = os.path.join(base_folder, '01_data', '01_preprocessed', 'ECHR_dict.pkl' )

#%% Initialize variables

doc_pars = []
articles = {}
tags = ['p', 'h4']
art_body = ''
extracting = False
invalid_strings = [None, '\n', '\n\n']
items_to_remove = ['', '*']

#%% Load data

parser = etree.HTMLParser(recover = True, encoding = 'utf-8')
tree = etree.parse(input_path, parser = parser)
root = tree.getroot()

#%% Retrieve and print headings (article number) and paragraphs (article text)

iterator = tree.iter(tag = tags)
for idx, elem in enumerate(iterator):
    if elem.tag == 'p' and elem.text not in invalid_strings:
        text = elem.text.strip().strip('\n')
        #print(text, '\n')
        doc_pars.append(text)
    elif elem.tag == 'h4':
        children = elem.getchildren()
        for child in children:
            if child.text != None:      
                text = child.text.strip().strip('\n')
                #print(text, '\n')
                doc_pars.append(text)

#%% Clean paragraphs

doc_pars = [x for x in doc_pars if x not in items_to_remove]
doc_pars = doc_pars[86:] # Remove initial list of articles

#%% Extract articles

for idx, par in enumerate(doc_pars):
    if par.startswith('Article') and extracting == False:
        extracting = True
        heading = par if par not in articles.keys() else (par + '99' + str(idx)) # Assing index + '99' to duplicated pars
    elif par.startswith('Article') and extracting == True:
        # Write article to dict
        if art_body != '':
            articles[heading] = art_body.strip()    
        # Initialize article variables
        heading = par if par not in articles.keys() else (par + '99' + str(idx)) # Assing index + '99' to duplicated pars
        art_body = ''
    else:
        art_body += ' ' + par

articles = {int(key.split(' ')[1]):value for key, value in articles.items()}

#%% Retrieve and write to file headings (article number) and paragraphs (article text)

with open (output_path, 'wb') as fw:
    pickle.dump(articles, fw)
