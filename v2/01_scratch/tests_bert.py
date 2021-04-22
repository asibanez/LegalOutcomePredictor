from transformers import AutoTokenizer, AutoModel

model_name = 'nlpaueb/legal-bert-small-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
bert_model = AutoModel.from_pretrained(model_name)

facts = ['hey, how are you baby?', 'my name is Tony']
facts_tokens = bert_tokenizer(facts, return_tensors = 'pt', padding = True)

bert_model(**facts_tokens)
