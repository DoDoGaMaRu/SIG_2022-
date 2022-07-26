import json
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

print("tokenized_data open...")
with open("../data/token/tokenized_data.json", 'r', encoding='utf-8') as f:
    tokenized_data = json.loads(f.read())
    del f
print("complete")


print("make train_data")
train_data = []
for lyrics in tqdm(tokenized_data):
    train_data += lyrics

del tokenized_data


from gensim.models import Word2Vec
print('make model...')
model = Word2Vec(sentences = train_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print('save model...')
model.wv.save_word2vec_format('../data/model/music_w2v')