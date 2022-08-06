import sys
import itertools
import numpy as np

from numpy import (
    dot, float16 as REAL, double, zeros, vstack, ndarray,
    sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

from gensim import utils


class LyricsDB():
    def __init__(self, vector_size, count=0, dtype=np.float16, mapfile_path=None):
        self.vector_size = vector_size
        self.index_to_key = [None] * count  # fka index2entity or index2word
        self.next_index = 0  # pointer to where next new entry will land
        self.key_to_index = {}

        self.vectors = zeros((count, vector_size), dtype=dtype)  # formerly known as syn0
        self.norms = None

        # "expandos" are extra attributes stored for each key: {attribute_name} => numpy array of values of
        # this attribute, with one array value for each vector key.
        # The same information used to be stored in a structure called Vocab in Gensim <4.0.0, but
        # with different indexing: {vector key} => Vocab object containing all attributes for the given vector key.
        #
        # Don't modify expandos directly; call set_vecattr()/get_vecattr() instead.
        self.expandos = {}

        self.mapfile_path = mapfile_path

    def __str__(self):
        return f"{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>"

    @classmethod
    def load_word2vec_format(
            cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
            limit=None, datatype=REAL, no_header=False,
    ):
        return _load_word2vec_format(
            cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype, no_header=no_header,
        )


def _add_word_to_kv(kv, counts, word, weights, vocab_size):

    if kv.has_index_for(word):
        return
    word_id = kv.add_vector(word, weights)

    if counts is None:
        # Most common scenario: no vocab file given. Just make up some bogus counts, in descending order.
        # TODO (someday): make this faking optional, include more realistic (Zipf-based) fake numbers.
        word_count = vocab_size - word_id
    elif word in counts:
        # use count from the vocab file
        word_count = counts[word]
    else:
        word_count = None
    kv.set_vecattr(word, 'count', word_count)

def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index  # don't read more than kv preallocated to hold
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1

        if i_space == -1 or (len(chunk) - i_vector) < bytes_per_vector:
            break

        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        # Some binary files are reported to have obsolete new line in the beginning of word, remove it
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1

    return processed_words, chunk[start:]


def _word2vec_read_binary(
        fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size,
        encoding="utf-8",
):
    chunk = b''
    tot_processed_words = 0

    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        processed_words, chunk = _add_bytes_to_kv(
            kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")


def _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding):
    for line_no in range(vocab_size):
        line = fin.readline()
        if line == b'':
            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        _add_word_to_kv(kv, counts, word, weights, vocab_size)


def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
    word, weights = parts[0], [datatype(x) for x in parts[1:]]
    return word, weights


def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:  # EOF/max: return what we've got
            break
        if vector_size:
            continue  # don't bother parsing lines past the 1st
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return vocab_size, vector_size

def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:  # EOF/max: return what we've got
            break
        if vector_size:
            continue  # don't bother parsing lines past the 1st
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return vocab_size, vector_size

def _load_word2vec_format(
        cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
        limit=sys.maxsize, datatype=REAL, no_header=False, binary_chunk_size=100 * 1024,
):

    counts = None
    if fvocab is not None:
        counts = {}
        with utils.open(fvocab, 'rb') as fin:
            for line in fin:
                word, count = utils.to_unicode(line, errors=unicode_errors).strip().split()
                counts[word] = int(count)

    with utils.open(fname, 'rb') as fin:
        if no_header:
            # deduce both vocab_size & vector_size from 1st pass over file
            if binary:
                raise NotImplementedError("no_header only available for text-format files")
            else:  # text
                vocab_size, vector_size = _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding)
            fin.close()
            fin = utils.open(fname, 'rb')
        else:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = [int(x) for x in header.split()]  # throws for invalid file format
        if limit:
            vocab_size = min(vocab_size, limit)
        kv = cls(vector_size, vocab_size, dtype=datatype)

        if binary:
            _word2vec_read_binary(
                fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding
            )
        else:
            _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding)
    if kv.vectors.shape[0] != len(kv):
        kv.vectors = ascontiguousarray(kv.vectors[: len(kv)])
    assert (len(kv), vector_size) == kv.vectors.shape

    kv.add_lifecycle_event(
        "load_word2vec_format",
        msg=f"loaded {kv.vectors.shape} matrix of type {kv.vectors.dtype} from {fname}",
        binary=binary, encoding=encoding,
    )
    return kv