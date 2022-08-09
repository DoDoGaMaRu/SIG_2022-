from gensim.models import KeyedVectors
import numpy as np

class Sent2vec:
    def __init__(self, w2v_path):
        print("\tload w2v model...")
        self.w2v_kv = self.load_w2v_model(w2v_path)
        self.embedding_dim = self.w2v_kv.vectors.shape[1]
        self.ZERO_VEC = np.zeros(self.embedding_dim, dtype=np.float16)

    def load_w2v_model(self, w2v_path):
        return KeyedVectors.load_word2vec_format(w2v_path)


    def convert_16bit(self, vec):
        return np.array(vec, dtype=np.float16)

    def sent2vec(self, tokenized_sentence):
        size = len(tokenized_sentence)
        if size < 1:
            return self.ZERO_VEC

        sum_of_vec = 0.
        for word in tokenized_sentence:
            current_vec = self.get_word_vector(word)
            sum_of_vec += current_vec
            
        try :
            senVec = sum_of_vec / size
        except ZeroDivisionError:
            senVec = self.ZERO_VEC

        return self.convert_16bit(senVec)


    def get_word_vector(self, word):
        try :
            vector = self.w2v_kv[word]
        except KeyError:
            vector = self.ZERO_VEC

        return vector


    def get_zero_vec(self):
        return self.ZERO_VEC

    def get_dim(self):
        return self.embedding_dim