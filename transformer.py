from array import array
from numbers import Number
from typing import List
import numpy as np
import scipy.sparse as sp
class Transformer:
    def __init__(self,unkmarker='*'):
        self.feature_names_ = []
        self.vocabulary_ = {}
        self.separator = '='
        self.dtype = np.float64
        self.unkmarker = unkmarker
    def generateMapping(self,data):
        feature_names = []
        vocab = {}
        for x in data:
            for f, v in x.items():
                if isinstance(v, str):
                    feature_name = "%s%s%s" % (f, self.separator, v)
                    v = 1
                elif isinstance(v, Number) or (v is None):
                    feature_name = f
                if feature_name is not None:
                    if feature_name not in vocab:
                        vocab[feature_name] = len(feature_names)
                        feature_names.append(feature_name)

        # feature_names.sort()
        # vocab = {f: i for i, f in enumerate(feature_names)}

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab
        self.vocabulary_[self.unkmarker] = len(feature_names)
        self.feature_names_.append(self.unkmarker)
        return self

    def map(self,X):
        dtype = self.dtype
        
        vocab = self.vocabulary_
        # Process everything as sparse regardless of setting

        indices = array("i")
        indptr = [0]
        # XXX we could change values to an array.array as well, but it
        # would require (heuristic) conversion of dtype to typecode...
        values = []

        # collect all the possible feature names and build sparse matrix at
        # same time
        for x in X:
            for f, v in x.items():
                if isinstance(v, str):
                    feature_name = "%s%s%s" % (f, self.separator, v)
                    v = 1
                elif isinstance(v, Number) or (v is None):
                    feature_name = f
                if feature_name is not None:
                    if feature_name in vocab:
                        indices.append(vocab[feature_name])
                        values.append(self.dtype(v))
                    else:
                        indices.append(vocab[self.feature_names_[-1]])
                        values.append(1)
            indptr.append(len(indices))

        if len(indptr) == 1:
            raise ValueError("Sample sequence X is empty.")

        indices = np.frombuffer(indices, dtype=np.intc)
        shape = (len(indptr) - 1, len(vocab))

        result_matrix = sp.csr_matrix(
            (values, indices, indptr), shape=shape, dtype=dtype
        )

        # Sort everything if asked
        result_matrix = result_matrix.toarray()
        return result_matrix