from transformers import AutoTokenizer, AutoModel

model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
bert_model = AutoModel.from_pretrained(model_name)

facts = ['hey, how are you baby?', 'my name is Tony']
facts_tokens = bert_tokenizer(facts, return_tensors = 'pt', padding = True)

bert_model(**facts_tokens)


####-------------------------------------------------------------------------

from transformers import AutoModel
from transformers import AutoTokenizer

model_name = 'nlpaueb/legal-bert-small-uncased'
bert_model = AutoModel.from_pretrained(model_name)

#%%
for parameter in bert_model.parameters():
    print(parameter)
#%% 
for parameter in bert_model.parameters():
    parameter.requires_grad = False
#%%
for parameter in bert_model.parameters():
    print(parameter)


 
