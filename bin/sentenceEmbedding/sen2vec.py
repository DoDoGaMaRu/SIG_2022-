from gensim.models import KeyedVectors
import numpy as np

print("[sen2vec]")
print("\tload model...")
model = KeyedVectors.load_word2vec_format("../../data/model/music_w2v.model") #C:\Users\백대환\Desktop\IdeaProjects\SIG_2022하계\data\model\music_w2v
print("\tcomplete")

embedding_dim = model.vectors.shape[1]
ZERO_VEC = np.zeros(embedding_dim)

def sen2vec(tokenized_sentence):
    size = len(tokenized_sentence)


    sumOfVec = 0
    for word in tokenized_sentence :
        try :
            sumOfVec += model[word]
            print(model[word])
        except KeyError:
            sumOfVec += ZERO_VEC

    try :
        senVec = sumOfVec / size
    except ZeroDivisionError:
        senVec = ZERO_VEC

    return senVec