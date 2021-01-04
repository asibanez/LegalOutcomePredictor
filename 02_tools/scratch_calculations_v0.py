import nltk
from matplotlib import pyplot as plt

#%%
lens_list = []

for key in articles.keys():
    tokens = nltk.word_tokenize(articles[key])
    lens_list.append(len(tokens))

max_len = max(lens_list)
avg_len = sum(lens_list) / len(lens_list)

print(f'max length = {max_len}')
print(f'avg length = {avg_len}')

#%%
plt.hist(lens_list)
plt.xlabel('# tokens')
plt.ylabel('freq')
plt.show()

