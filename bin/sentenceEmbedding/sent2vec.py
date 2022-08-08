from gensim.models import KeyedVectors
import numpy as np

class Sent2vec:
    def __init__(self, w2v_path):
        print("\tload w2v model...")
        self.w2v_kv = self.load_w2v_model(w2v_path)
        self.embedding_dim = self.w2v_kv.vectors.shape[1]
        self.ZERO_VEC = np.zeros(self.embedding_dim)

    def load_w2v_model(self, w2v_path):
        return KeyedVectors.load_word2vec_format(w2v_path)


    def convert16bit(self, vec):
        return np.array(vec, dtype=np.float16)

    def sent2vec(self, tokenized_sentence):
        size = len(tokenized_sentence)

        sumOfVec = 0
        for word in tokenized_sentence :
            try :
                sumOfVec += self.w2v_kv[word]
            except KeyError:
                sumOfVec += self.ZERO_VEC

        try :
            senVec = sumOfVec / size
        except ZeroDivisionError:
            senVec = self.ZERO_VEC

        return self.convert16bit(senVec)


    def getZeroVec(self):
        return self.ZERO_VEC

    def getDim(self):
        return self.embedding_dim