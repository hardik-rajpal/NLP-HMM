from array import array
from numbers import Number
from typing import List
import numpy as np
import scipy.sparse as sp

from nltk.corpus import treebank,brown
CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

sents = treebank.tagged_sents(tagset='universal')
k = int(len(sents)/10)
# print(k)
# exit()
sents = sents[:k]
tags = set([
    tag for sentence in sents
    for _,tag in sentence
])
train_test_cutoff= int(0.8*len(sents))
trainsents = sents[:train_test_cutoff]
testsents = sents[train_test_cutoff:]

train_val_cutoff = int(0.25*len(trainsents))
valsents = trainsents[:train_val_cutoff]
trainsents = trainsents[train_val_cutoff:]
def add_basic_features(sentence_terms:List[str],index:int):
    term = sentence_terms[index]
    return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'is_capitalized': term[0].upper() == term[0],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'prefix-1': term[0],
        'prefix-2': term[:2],
        'prefix-3': term[:3],
        'suffix-1': term[-1],
        'suffix-2': term[-2:],
        'suffix-3': term[-3:],
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
    }
def untag(taggedsents):
    return [w for w,_ in taggedsents]
def transformToDataset(taggedsents):
    x,y = [],[]
    for postags in taggedsents:
        for index, (term,class_) in enumerate(postags):
            x.append(add_basic_features(untag(postags),index))
            y.append(class_)
    return x,y
xtrain, ytrain = transformToDataset(trainsents)
xtest,ytest = transformToDataset(testsents)
xval,yval = transformToDataset(valsents)
class Transformer:
    def __init__(self):
        self.feature_names_ = []
        self.vocabulary_ = {}
        self.separator = '='
        self.dtype = np.float64
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

dict_vectorizer = Transformer()
dict_vectorizer.generateMapping(xtrain+xtest+xval)
xtrain = dict_vectorizer.map(xtrain)
xtest = dict_vectorizer.map(xtest)
xval = dict_vectorizer.map(xval)
# print(xtrain[0,:10])
# exit()
# from sklearn.preprocessing import LabelEncoder
# labelEncoder = LabelEncoder()
# labelEncoder.fit(ytrain+ytest+yval)
# ytrain = labelEncoder.transform(ytrain)
# ytest = labelEncoder.transform(ytest)
# yval = labelEncoder.transform(yval)
# from keras.utils import np_utils
# ytrain = np_utils.to_categorical(ytrain,num_classes=len(tags))
# ytest = np_utils.to_categorical(ytest,num_classes=len(tags))
# yval = np_utils.to_categorical(yval,num_classes=len(tags))
# print(ytrain.shape,ytest.shape,yval.shape)
# exit()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
def build_model(input_dim, hidden_neurons, output_dim):
    """
    Construct, compile and return a Keras model which will be used to fit/predict
    """
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],)
    return model
from keras.wrappers.scikit_learn import KerasClassifier

modelparams = {
    'build_fn': build_model,
    'input_dim': 17443,
    'hidden_neurons': 512,
    'output_dim':12,
    'epochs': 5,
    'batch_size': 256,
    'verbose': 1,
    # 'validation_data': (xval, yval),
    'shuffle': True
}
clf = KerasClassifier(**modelparams)
if __name__=='__main__':
    hist = clf.fit(xtrain,ytrain)

    
    score = clf.score(xtest,ytest)
    print(score)