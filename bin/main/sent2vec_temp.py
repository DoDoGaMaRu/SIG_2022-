import sys
import itertools
import numpy as np


from numpy import (
    dot, float16 as REAL, double, zeros, vstack, ndarray,
    sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)

from gensim import utils, matutils
from numbers import Integral

_KEY_TYPES = (str, int, np.integer)
_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)


def _ensure_list(value):    #TODO 값 확인하고 형식 맞춰주는듯
    if value is None:
        return []

    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]

    if isinstance(value, ndarray) and len(value.shape) == 2:
        return list(value)

    return value



class vectorFinder(utils.SaveLoad):
    def __init__(self, vector_size, count=0, dtype=np.float16, mapfile_path=None):
        self.vector_size = vector_size
        self.index_to_key = [None] * count
        self.next_index = 0
        self.key_to_index = {}

        self.vectors = zeros((count, vector_size), dtype=dtype)
        self.norms = None
        self.expandos = {}

        self.mapfile_path = mapfile_path

    def __str__(self):
        return f"{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>"

    def __len__(self):
        return len(self.index_to_key)

    def __getitem__(self, key_or_keys):
        if isinstance(key_or_keys, _KEY_TYPES):
            return self.get_vector(key_or_keys)


    def get_index(self, key, default=None):
        """Return the integer index (slot/position) where the given key's vector is stored in the
        backing vectors array.

        """
        val = self.key_to_index.get(key, -1)
        if val >= 0:
            return val
        elif isinstance(key, (int, np.integer)) and 0 <= key < len(self.index_to_key):
            return key
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key '{key}' not present")

    def has_index_for(self, key):
        """Can this model return a single index for this key?

        Subclasses that synthesize vectors for out-of-vocabulary words (like
        :class:`~gensim.models.fasttext.FastText`) may respond True for a
        simple `word in wv` (`__contains__()`) check but False for this
        more-specific check.

        """
        return self.get_index(key, -1) >= 0



    def add_vector(self, key, vector):
        """Add one new vector at the given key, into existing slot if available.

        Warning: using this repeatedly is inefficient, requiring a full reallocation & copy,
        if this instance hasn't been preallocated to be ready for such incremental additions.

        Parameters
        ----------

        key: str
            Key identifier of the added vector.
        vector: numpy.ndarray
            1D numpy array with the vector values.

        Returns
        -------
        int
            Index of the newly added vector, so that ``self.vectors[result] == vector`` and
            ``self.index_to_key[result] == key``.

        """
        target_index = self.next_index
        if target_index >= len(self) or self.index_to_key[target_index] is not None:
            # must append at end by expanding existing structures
            target_index = len(self)
            self.add_vectors([key], [vector])
            self.allocate_vecattrs()  # grow any adjunct arrays
            self.next_index = target_index + 1
        else:
            # can add to existing slot
            self.index_to_key[target_index] = key
            self.key_to_index[key] = target_index
            self.vectors[target_index] = vector
            self.next_index += 1
        return target_index



    def set_vecattr(self, key, attr, val):
        """Set attribute associated with the given key to value.

        Parameters
        ----------

        key : str
            Store the attribute for this vector key.
        attr : str
            Name of the additional attribute to store for the given key.
        val : object
            Value of the additional attribute to store for the given key.

        Returns
        -------

        None

        """
        self.allocate_vecattrs(attrs=[attr], types=[type(val)])
        index = self.get_index(key)
        self.expandos[attr][index] = val


    def allocate_vecattrs(self, attrs=None, types=None):
        """Ensure arrays for given per-vector extra-attribute names & types exist, at right size.

        The length of the index_to_key list is canonical 'intended size' of KeyedVectors,
        even if other properties (vectors array) hasn't yet been allocated or expanded.
        So this allocation targets that size.

        """
        # with no arguments, adjust lengths of existing vecattr arrays to match length of index_to_key
        if attrs is None:
            attrs = list(self.expandos.keys())
            types = [self.expandos[attr].dtype for attr in attrs]
        target_size = len(self.index_to_key)
        for attr, t in zip(attrs, types):
            if t is int:
                t = np.int64  # ensure 'int' type 64-bit (numpy-on-Windows https://github.com/numpy/numpy/issues/9464)
            if t is str:
                # Avoid typing numpy arrays as strings, because numpy would use its fixed-width `dtype=np.str_`
                # dtype, which uses too much memory!
                t = object
            if attr not in self.expandos:
                self.expandos[attr] = np.zeros(target_size, dtype=t)
                continue
            prev_expando = self.expandos[attr]
            if not np.issubdtype(t, prev_expando.dtype):
                raise TypeError(
                    f"Can't allocate type {t} for attribute {attr}, "
                    f"conflicts with its existing type {prev_expando.dtype}"
                )
            if len(prev_expando) == target_size:
                continue  # no resizing necessary
            prev_count = len(prev_expando)
            self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
            self.expandos[attr][: min(prev_count, target_size), ] = prev_expando[: min(prev_count, target_size), ]



    def fill_norms(self, force=False):      #TODO 놈유효성 판단
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    def get_mean_vector(self, keys, weights=None, pre_normalize=True, post_normalize=False, ignore_missing=True):
        """Get the mean vector for a given list of keys.

        Parameters
        ----------

        keys : list of (str or int or ndarray)
            Keys specified by string or int ids or numpy array.
        weights : list of float or numpy.ndarray, optional
            1D array of same size of `keys` specifying the weight for each key.
        pre_normalize : bool, optional
            Flag indicating whether to normalize each keyvector before taking mean.
            If False, individual keyvector will not be normalized.
        post_normalize: bool, optional
            Flag indicating whether to normalize the final mean vector.
            If True, normalized mean vector will be return.
        ignore_missing : bool, optional
            If False, will raise error if a key doesn't exist in vocabulary.

        Returns
        -------

        numpy.ndarray
            Mean vector for the list of keys.

        Raises
        ------

        ValueError
            If the size of the list of `keys` and `weights` doesn't match.
        KeyError
            If any of the key doesn't exist in vocabulary and `ignore_missing` is false.

        """
        if len(keys) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(weights, list):
            weights = np.array(weights)
        if weights is None:
            weights = np.ones(len(keys))
        if len(keys) != weights.shape[0]:  # weights is a 1-D numpy array
            raise ValueError(
                "keys and weights array must have same number of elements"
            )

        mean = np.zeros(self.vector_size, self.vectors.dtype)

        total_weight = 0
        for idx, key in enumerate(keys):
            if isinstance(key, ndarray):
                mean += weights[idx] * key
                total_weight += abs(weights[idx])
            elif self.__contains__(key):
                vec = self.get_vector(key, norm=pre_normalize)
                mean += weights[idx] * vec
                total_weight += abs(weights[idx])
            elif not ignore_missing:
                raise KeyError(f"Key '{key}' not present in vocabulary")

        if(total_weight > 0):
            mean = mean / total_weight
        if post_normalize:
            mean = matutils.unitvec(mean).astype(REAL)
        return mean



    def most_similar(
            self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None,
            restrict_vocab=None, indexer=None,
        ):


        if isinstance(topn, Integral) and topn < 1: #topn이 숫자인지 확인
            return []

        # allow passing a single string-key or vector for the positive/negative arguments
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)           #TODO 필요없을거임

        self.fill_norms()                           #TODO 놈 유효성 판단 필요한지 모르겠음
        clip_end = clip_end or len(self.vectors)

        if restrict_vocab:
            clip_start = 0
            clip_end = restrict_vocab

        # add weights for each key, if not already present; default to 1.0 for positive and -1.0 for negative keys
        keys = []
        weight = np.concatenate((np.ones(len(positive)), -1.0 * np.ones(len(negative))))
        for idx, item in enumerate(positive + negative):
            if isinstance(item, _EXTENDED_KEY_TYPES):
                keys.append(item)
            else:
                keys.append(item[0])
                weight[idx] = item[1]

        # compute the weighted average of all keys
        mean = self.get_mean_vector(keys, weight, pre_normalize=True, post_normalize=True, ignore_missing=False)
        all_keys = [
            self.get_index(key) for key in keys if isinstance(key, _KEY_TYPES) and self.has_index_for(key)
        ]

        if indexer is not None and isinstance(topn, int):
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors[clip_start:clip_end], mean) / self.norms[clip_start:clip_end]
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_keys), reverse=True)
        # ignore (don't return) keys from the input
        result = [
            (self.index_to_key[sim + clip_start], float(dists[sim]))
            for sim in best if (sim + clip_start) not in all_keys
        ]
        return result[:topn]



    @classmethod
    def load_word2vec_format(      #자신 객체 생성하는 거인듯
            cls, fname, binary=False, encoding='utf8', unicode_errors='strict',
            limit=None, datatype=REAL, no_header=False,
    ):
        return _load_word2vec_format(
            cls, fname, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
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
        cls, fname, binary=False, encoding='utf8', unicode_errors='strict',
        limit=sys.maxsize, datatype=REAL, no_header=False, binary_chunk_size=100 * 1024,
    ):

    counts = None

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