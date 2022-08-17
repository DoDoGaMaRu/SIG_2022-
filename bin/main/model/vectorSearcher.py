import sys
import itertools
import numpy as np

from numpy import (
    dot, float16 as REAL, double, zeros, vstack, ndarray,
    sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

from gensim import utils, matutils
from numbers import Integral


class VectorSearcher(utils.SaveLoad):
    def __init__(self, vector_size, count=0, dtype=np.float16):
        self.vector_size = vector_size
        self.index_to_key = [None] * count
        self.next_index = 0

        self.vectors = zeros((count, vector_size), dtype=dtype)
        self.norms = None

    def __str__(self):
        return f"{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>"

    def __len__(self):
        return len(self.index_to_key)



    def add_vector(self, key, vector):
        target_index = self.next_index

        self.index_to_key[target_index] = key
        self.vectors[target_index] = vector
        self.next_index += 1

        return target_index

    def fill_norms(self):      # norm 만들기
        if self.norms is None:
            self.norms = np.linalg.norm(self.vectors, axis=1)


    def most_similar(self, positive_vec=None, topn=10):
        if len(positive_vec) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(topn, Integral) and topn < 1:
            return []

        self.fill_norms()
        mean = matutils.unitvec(positive_vec).astype(REAL)  # 정규화인데 필요한지 잘 모르겠음

        dists = dot(self.vectors, mean) / self.norms
        best = matutils.argsort(dists, topn=topn, reverse=True)

        result = [
            (self.index_to_key[sim], float(dists[sim])) for sim in best
        ]

        return result

    # TODO eucladean distance로 유사도 비교
    def dist(self, x, y):
        return np.sum((x-y)**2)

    def most_similar_euclidean(self, positive_vec=None, topn=10):
        if len(positive_vec) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(topn, Integral) and topn < 1:
            return []

        dists = [self.dist(vec, positive_vec) for vec in self.vectors]
        best = matutils.argsort(dists, topn=topn)

        result = [
            (self.index_to_key[sim], float(dists[sim])) for sim in best
        ]

        return result



    @classmethod
    def load_word2vec_format(
            cls, fname, encoding='utf8', unicode_errors='strict', datatype=REAL
    ):
        print("[sentence vector data]")
        print("\tload data...")
        return _load_word2vec_format(
            cls, fname, encoding=encoding, unicode_errors=unicode_errors, datatype=datatype,
        )

def _load_word2vec_format(cls, fname, encoding='utf8', unicode_errors='strict', datatype=REAL, binary_chunk_size=100 * 1024):
    with utils.open(fname, 'rb') as fin:
        header = utils.to_unicode(fin.readline(), encoding=encoding)    # get vocab, vector size
        vocab_size, vector_size = [int(x) for x in header.split()]

        vocab_size = min(vocab_size, sys.maxsize)
        kv = cls(vector_size, vocab_size, dtype=datatype)

        _word2vec_read_text(fin, kv, vocab_size, datatype, unicode_errors, encoding)

    return kv


def _word2vec_read_text(fin, kv, vocab_size, datatype, unicode_errors, encoding):
    for line_no in range(vocab_size):
        line = fin.readline()
        if line == b'':     #바이트일경우 에러
            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
        element, vector = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        kv.add_vector(element, vector)


def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
    element, vector = parts[0], [datatype(x) for x in parts[1:]]
    return element, vector