# Imports
import os
from lxml import etree

#%% Define paths
data_folder = 'C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\01_Local\\05_Liberty_Mutual\\Research\\02_legal_outcome_predictor\\00_data'
input_path = os.path.join(data_folder, '00_law\\00_original\\EUCHR_html\\ECHR.html')

#%% Load data
parser = etree.HTMLParser(recover = True, encoding = 'utf-8')
tree = etree.parse(input_path, parser = parser)
root = tree.getroot()

#%% Print whole document
print(etree.tostring(root, pretty_print = True, method = 'html').decode())

#%% Print all elements in tree
for element in tree.iter(): print(element)

#%% Print all elements in element
for element in root.iter(): print(element)

#%% Retrieve element from root and show values
element = list(root.iter())[160]
print('Tag =', element.tag, '\n')
print('Text =', element.text, '\n')
print('Attributes = ', element.attrib, '\n')

#%% Get root children
root_children_1 = root.getchildren()
root_children_2 = list(root)
print(root_children_1)

#%% Get all <p>
for item in tree.findall('.//p'):
    print(item.text)
