from functools import partial
from time import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import re
from keras.utils import np_utils,plot_model
from nltk.corpus import brown,treebank
import torch.nn as nn 
import torch
from transformer import Transformer
from kerastagger import clf
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
import tensorflow as tf
tf.get_logger().setLevel(5)


"""
Train Order:
1. get corpus from nltk. list[list[tuple[str]]]
2. get words, tags separated into lists. list[list[str]], list[list[str]]
3. tags => taginds. list[list[int]] => tagvectors list[list[list[int]]]
4. words => word-feature-vectors. list[list[list[float]]]
5. train model.
"""
AMT = '$amt$'
QNW = '$qnw$'
UNKW = '*'
###Model related data:
SEED_VALUE = 4
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
MAXPSLENGTH = 3
hiddenLayers = [600,400]
MAX_EPOCHS = 5
NUMTAGS = 12
BATCHSZ = 256
LOSSFN = 'CEL'
MINETYLEN = 5
lr = 0.01
def getModel(layerszs):
    model = Sequential([
        Dense(layerszs[1], input_dim=layerszs[0]),
        Activation('relu'),
        Dropout(0.2),
    ])
    for i in range(2,len(layerszs)-1):
        model.add(Dense(layerszs[i]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(layerszs[-1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],)
    return model
###Fine tuning detes:
"""
(loss = TSS)
BS, layrszs => acc of saturation on train set.
[]=> 29% 
[24]=> max acc = 38, most values are 29%
256, [48]=> max acc = ...
(loss = CEL)

[]=>15% ... then 29%.

"""
### 

class Network():
    def __init__(self, lyrszs:List[int],useCEL,valX,valY):
        modelparams = {
            'build_fn': partial(getModel,lyrszs),
            'epochs': MAX_EPOCHS,
            'batch_size': BATCHSZ,
            'verbose': 1,
            'validation_data': (valX,valY),
            'shuffle': True
        }
        self.clf = KerasClassifier(**modelparams)

    def accuracy(self,predvects:torch.Tensor,labels:torch.Tensor):
        np.savetxt('preds.csv',predvects.detach().numpy())
        np.savetxt('labsel.txt',labels.numpy(),fmt='%d')
        return np.round(100*(labels[torch.argmax(predvects,1)==torch.argmax(labels,1)].shape[0]/labels.shape[0]),2)
    def plotmodel(self):
        plot_model(self.clf.model, to_file='modelArch.png', show_shapes=True)
class Tagger:
    isAmount = lambda word:(re.compile('\W[\d,.]+').match(word)!=None)
    isQualNum = lambda word:(re.compile('[\d\w,-/]*\d[\d\w,-/]*').match(word)!=None)
    def __init__(self):
        self.tags = []
        self.taginds = {}#dictionary mapping tags to indices in tags.
        self.prefixes = np.loadtxt('prefs.txt',dtype=np.unicode_).tolist()
        self.suffixes = np.loadtxt('suffs.txt',dtype=np.unicode_).tolist()
        self.prefixes = sorted(self.prefixes,key=lambda s:-len(s))
        self.suffixes = sorted(self.suffixes,key=lambda s:-len(s))
        self.prefinds = {}
        self.suffinds = {}
        self.rawsuffs = set()
        self.rawprefs = set()
        self.rawprefinds = {}
        self.rawsuffinds = {}
        self.rawsuffslen = 0
        self.rawprefslen = 0
        for i,pref in enumerate(self.prefixes):
            self.prefinds[pref] = i
        for i,suff in enumerate(self.suffixes):
            self.suffinds[suff] = i
        self.model = None # set to a nn.Module object after training.
        self.transformer = None # set to a Transformer after setupTransformer()
    def evalMetrics(self,preds,trueTags):
        #unpack all preds,tags sentences into one array:
        for i in range(1,len(preds)):
            preds[0].extend(preds[i])
            trueTags[0].extend(trueTags[i])
        preds = preds[0]
        trueTags = trueTags[0]
        ##unpacking over.
        numtags = len(self.tags)
        confmat = np.zeros((numtags,numtags))
        total = len(preds)

        for pred,trueTag in zip(preds,trueTags):
            if(self.taginds.__contains__(trueTag)):
                trueTag = self.taginds[trueTag]
                confmat[pred,trueTag]+=1
            else:
                print(trueTag)
        accuracy = np.sum(np.array([confmat[i,i] for i in range(numtags)]))/total
        ppos = []
        pposr = [confmat[i,i]/(max(1,np.sum(confmat[:,i]))) for i in np.arange(numtags)]
        pposp = [confmat[i,i]/(max(1,np.sum(confmat[i,:]))) for i in np.arange(numtags)]
        pposf1 = [((2*pposp[i]*pposr[i])/(1e-15+(pposp[i])+(pposr[i]))) for i in np.arange(numtags)]
        ppos.append(pposp)
        ppos.append(pposr)
        ppos.append(pposf1)
        return confmat,accuracy,np.array(ppos)
    def prefindex(self,word:str,numprefs:int):
        prefs = []
        if(not (self.isnumeric(word) or len(word)<MINETYLEN)):
            for i in range(1,len(word)):
                if(self.prefinds.__contains__(word[:i])):
                    prefs.append(self.prefinds[word[:i]])
                    if(len(prefs)==numprefs):
                        break
        while(len(prefs)<numprefs):
            prefs.append(-1)
        return prefs
    def suffindex(self,word:str,numsuffs:int):
        suffs = []
        if(not (self.isnumeric(word) or len(word)<MINETYLEN)):
            for i in range(1,len(word)):
                if(self.suffinds.__contains__(word[-i:])):
                    suffs.append(self.suffinds[word[-i:]])
                    if(len(suffs)==numsuffs):
                        break
        while(len(suffs)<numsuffs):
            suffs.append(0)
        return suffs
    def isnumeric(self,word):
        isnumeric = 1 if word == AMT or word == QNW else 0
        return isnumeric
    def rawprefindex(self,word:str,numprefs):
        rawprefs = []
        for i in range(min(len(word),numprefs)):
            pref = word[:(1+i)]
            rawprefs.append(pref)
        while(len(rawprefs)<numprefs):
            rawprefs.append('')
        return rawprefs
    def rawsuffindex(self,word:str,numsuffs):
        rawsuffs = []
        for i in range(min(len(word),numsuffs)):
            suff = word[-(1+i):]
            rawsuffs.append(suff)
        while(len(rawsuffs)<numsuffs):
            rawsuffs.append('')
        return rawsuffs
    def featuresOf(self,j,sent,word:str):
        prevword = '' if j==0 else sent[j-1]
        nextword = '' if j+1==len(sent) else sent[j+1]
        isnumeric = self.isnumeric(sent[j])
        iscapitalized = 0
        isallcapitalized = 0
        isalllower = -1
        if isnumeric == 0:
            if ((word[0]!=word[0].lower())):
                iscapitalized = 1
            if(word.upper()==word):
                isallcapitalized = 1
            if(word.lower()==word):
                isalllower = 1
            else:
                isalllower = 0
        prefs = self.rawprefindex(sent[j],MAXPSLENGTH)
        suffs = self.rawsuffindex(sent[j],MAXPSLENGTH)
        feats =  ({
            'numwords':len(sent),#length of sentence
            'word':sent[j], #word
            'isFirst':1 if j==0 else 0,#is_first
            'isLast':1 if j==len(sent)-1 else 0,#is_last
            'isCapitalized':iscapitalized,#iscapitalized
            'isAllCapitalized':isallcapitalized,#isallcapitalized,
            'isAllLower':isalllower,
            'prevWord':prevword,#prev_word
            'nextWord':nextword,#next_word
            #suffix-index
            # isnumeric,#isnumeric
            # -2 if j==0 else self.prefindex(sent[j-1]),
            # -2 if j==len(sent)-1 else self.suffindex(sent[j+1]),
            # *(tagprobs.tolist())
        })
        for i in range(MAXPSLENGTH):
            feats[f'pref-{i}'] = prefs[i]
            feats[f'suff-{i}'] = suffs[i]
        return feats
    def trainNetwork(self,trainX:np.ndarray,trainY:np.ndarray,valX:np.ndarray,valY:np.ndarray):
        self.network = Network([trainX.shape[1],*hiddenLayers,trainY.shape[1]],LOSSFN=='CEL',valX,valY)
        self.model:KerasClassifier = self.network.clf
        self.model.fit(trainX,trainY)
        return self.model      
    def separateWordsTags(self,mapped_sents):
        words:list[list[str]] = []
        tags:list[list[str]] = []
        for sent in mapped_sents:
            words.append([]);tags.append([])
            for (word,tag) in sent:
                words[-1].append(word)
                tags[-1].append(tag)
        return words,tags
    def mapTagsToInds(self,tags):
        tagset = set()
        for tagsent in tags:
            for tag in tagsent:
                tagset.add(tag)
            if(len(tagset)==NUMTAGS):
                break
        tagset = list(tagset)
        tagset.append('X')
        tagset = sorted(tagset)
        self.tags = tagset
        self.taginds.clear()
        for i,tag in enumerate(tagset):
            self.taginds[tag] = i
        mappedTags = []
        for tagsent in tags:
            mappedTags.append([])
            for i in range(len(tagsent)):
                mappedTags[-1].append(self.taginds[tagsent[i]])
        return mappedTags
    def setupTransformer(self,trainWordFeats):
        self.transformer = Transformer()
        self.transformer.generateMapping(trainWordFeats)
        return self.transformer
    def mapWordsToFVsTrain(self,words,isTrain=True):
        preprocwords = self.preProcSents(words,isTrain)
        # preprocwords = words
        featvects = []
        for i in range(len(preprocwords)):
            featvects.append([])
            for j in range(len(preprocwords[i])):
                # prevtag = -1 if j==0 else tags[i][j-1]
                featvects[-1].append(self.featuresOf(j,preprocwords[i],words[i][j]))
        return featvects
    def getTrainableData(self, trainSents):
        words,tags = self.separateWordsTags(trainSents)
        taginds = self.mapTagsToInds(tags)
        featwords = self.mapWordsToFVsTrain(words,True)
        collapsed_featwords = []
        collapsed_taginds = []
        print("Mapped words and tags to features.")
        for i in range(len(featwords)):
            collapsed_featwords.extend(featwords[i])
            collapsed_taginds.extend(taginds[i])
        _,counts = np.unique(collapsed_taginds,return_counts=True)
        np.savetxt('trained/tagfreqs.csv',counts,fmt='%d')
        self.setupTransformer(collapsed_featwords)
        npfeats = (self.transformer.map(collapsed_featwords))
        nptags = np.array(np_utils.to_categorical(collapsed_taginds,num_classes=NUMTAGS))
        return npfeats,nptags
    def trainOn(self,trainSents):
        npfeats,nptags = self.getTrainableData(trainSents)
        print(f"npfeats secured. shapes: {npfeats.shape}, {nptags.shape}")
        trainvalcutoff = int(4*npfeats.shape[0]/5)
        xtrain,ytrain = npfeats,nptags
        model = self.trainNetwork(xtrain[:trainvalcutoff],ytrain[:trainvalcutoff],xtrain[trainvalcutoff:],ytrain[trainvalcutoff:])
        self.model = model
    def testOn(self, testSents):
        featwords = self.mapWordsToFVsTrain(testSents,False)
        answer = []
        taglist = []
        for i in range(len(featwords)):
            taglist.append([])
            answer.append([])
            tagprobs = self.model.predict(self.transformer.map(featwords[i]))
            for j,(wi,ti) in enumerate(zip((testSents[i]),tagprobs)):
                w = testSents[i][j]
                t = self.tags[ti]
                answer[-1].append(f'{w}_{t}')
                taglist[-1].append(ti)
        return taglist,answer
    def preProcSents(self,sents, train):
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                word = sents[i][j]
                word = word.lower()
                if(Tagger.isAmount(word)):
                    word = AMT
                elif(Tagger.isQualNum(word)):
                    word = QNW
                sents[i][j] = word
        return sents
    def demoSent(self,sent):
        sent2 = sent[:].split(' ')
        tags = self.testOn([sent.split(' ')])[0]
        for i in range(len(tags)):
            sent2[i]+= '_'+tags[i]
        return ' '.join(sent2)
    def findEvalMetrics(kfold,fractionOfCorpus:int=0.005):
        sents = list(brown.tagged_sents(tagset="universal"))
        sents = sents[:int(len(sents)*fractionOfCorpus)]
        l = len(sents)
        perm = np.random.permutation(l)
        sents = np.asanyarray(sents,dtype=object)[perm].tolist()
        res = []
        now = time()
        allitersppos = []
        for i in range(kfold):
            p1 = sents[:int((i*l)/kfold)]
            p2 = sents[int((i+1)*l/kfold):]
            dummyTest = sents[int((-1*l)/kfold):]
            p1.extend(p2)
            trainSents = p1
            testSents = sents[int(i*l/kfold):int((i+1)*l/kfold)]
            # testSents = dummyTest
            testSentsOnlyWords = list(map(
                            lambda sent:list(map(lambda wt:wt[0],sent)),
                            testSents
                            ))
            testSentsOnlyTags = list(map(
                lambda sent:list(map(lambda wt:wt[1],sent)),
                testSents
            ))
            POStagger = Tagger()
            POStagger.trainOn(trainSents)
            predTags,answers = POStagger.testOn(testSentsOnlyWords)
            confmat, acc, pposa = POStagger.evalMetrics(predTags,testSentsOnlyTags)
            pposa*=100
            allitersppos.append(pposa)
            np.savetxt(f'results/pposa_{i+1}.csv',pposa,delimiter=', ',fmt='%.2f')
            np.savetxt(f'results/confmat_{i+1}.csv',confmat,fmt='%.2f')
            plt.imshow(confmat,cmap='hot',interpolation='nearest')
            plt.xticks(np.arange(NUMTAGS),labels=POStagger.tags,fontsize=8)
            plt.yticks(np.arange(NUMTAGS),labels=POStagger.tags,fontsize=8)
            plt.text(4,-1,f'Range:({np.round(np.min(confmat),2)},{np.round(np.max(confmat),2)})')
            plt.savefig(f'results/confmat_{i+1}.png')
            res.append(acc)
            print(f'Time for iteration {i+1}: {time() - now}')
            now = time()
            break
        POStagger.network.plotmodel()
        np.savetxt('results/perposmetrics.csv',np.mean(allitersppos,0),fmt='%.2f')
        np.savetxt('results/accuracy.csv',100*np.array(res),fmt='%.2f')
        return res
    def getTrainedTagger():
        t = Tagger()
        corpus = brown
        dataset = list(corpus.tagged_sents(tagset="universal"))
        trr = 0.006
        trainSents = dataset[:int(trr*len(dataset))]
        t.trainOn(trainSents)
        return t
if __name__=='__main__':
    # t = Tagger.getTrainedTagger()
    Tagger.findEvalMetrics(5)