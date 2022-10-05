from time import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import brown,treebank
import torch.nn as nn 
import torch
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
###Model related data:
SEED_VALUE = 4
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
MAXPSLENGTH = 3
hiddenLayers = [512,512]
MAX_EPOCHS = 5
NUMTAGS = 12
BATCHSZ = 256
LOSSFN = 'CEL'
MINETYLEN = 5
lr = 0.01

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
class Network(nn.Module):
    def __init__(self, lyrszs:List[int],useCEL):
        #lyrszs[0] = 12 (#features)
        #lyrszs[-1] = 12 (#tags)
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.useCEL = useCEL
        self.layers.append(nn.Linear(lyrszs[0],lyrszs[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(lyrszs[1],lyrszs[2]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        self.layers.append(nn.Linear(lyrszs[2],lyrszs[3]))
        self.layers.append(nn.Softmax(1))
        # for i in range(1,len(lyrszs)):
        #     self.layers.append(nn.Linear(lyrszs[i-1],lyrszs[i]))
        #     self.layers.append(nn.ReLU())
        # self.layers.append(nn.Softmax(1))
    def forward(self,X:List[int]):
        for layer in self.layers:
            X = layer(X)
        return X
    def loss(self,predvects,labels:torch.Tensor):
        labvects = torch.zeros_like(predvects)
        for x in list(zip(torch.tensor(np.arange(len(labels),dtype=int),dtype=torch.long),labels.type(torch.long))):
            labvects[x[0],x[1]] = 1
        if(self.useCEL):
            return nn.CrossEntropyLoss()(predvects,labvects)
        losses = (labvects-predvects)**2
        return torch.mean(torch.sum(losses,1))
    def accuracy(self,predvects:torch.Tensor,labels:torch.Tensor):
        np.savetxt('preds.csv',predvects.detach().numpy())
        np.savetxt('labsel.txt',labels.numpy(),fmt='%d')
        return np.round(100*(labels[torch.argmax(predvects,1)==labels].shape[0]/labels.shape[0]),2)
class Tagger:
    isAmount = lambda word:(re.compile('\W[\d,.]+').match(word)!=None)
    isQualNum = lambda word:(re.compile('[\d\w,-/]*\d[\d\w,-/]*').match(word)!=None)
    def __init__(self):
        self.tags = []
        self.words = []
        self.wordinds = {}#dictionary mapping words to indices in words.#helpful for mapping plural of a word to the same index if multiplexing nns and nn
        self.taginds = {}#dictionary mapping tags to indices in tags.
        self.init_freqs = []
        self.freqMap = {}
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
        self.emmis = None #set pre training
        self.model = None # set to a nn.Module object after training.
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
            pred = self.taginds[pred]
            trueTag = self.taginds[trueTag]
            confmat[pred,trueTag]+=1
        accuracy = np.sum(np.array([confmat[i,i] for i in range(numtags)]))/total
        ppos = []
        pposr = [confmat[i,i]/(max(1,np.sum(confmat[:,i]))) for i in np.arange(numtags)]
        pposp = [confmat[i,i]/(max(1,np.sum(confmat[i,:]))) for i in np.arange(numtags)]
        pposf1 = [((2*pposp[i]*pposr[i])/((pposp[i])+(pposr[i]))) for i in np.arange(numtags)]
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
    def rawprefindex(self,word:str,numprefs,onehot=True):
        rawprefs = []
        for i in range(min(len(word),numprefs)):
            pref = word[:(1+i)]
            if(self.rawprefinds.__contains__(pref)):
                rawprefs.append(self.rawprefinds[pref])
            else:
                rawprefs.append(self.rawprefinds['*'])
        while(len(rawprefs)<numprefs):
            rawprefs.append(self.rawprefinds[''])
        if(onehot):
            rawprefs = self.onehot(rawprefs)
        return rawprefs
    def rawsuffindex(self,word:str,numsuffs,onehot=True):
        rawsuffs = []
        for i in range(min(len(word),numsuffs)):
            suff = word[-(1+i):]
            if(self.rawsuffinds.__contains__(suff)):
                rawsuffs.append(self.rawsuffinds[suff])
            else:
                rawsuffs.append(self.rawsuffinds['*'])
        while(len(rawsuffs)<numsuffs):
            rawsuffs.append(self.rawsuffinds[''])
        if(onehot):
            rawsuffs = self.onehot(rawsuffs,True)
        return rawsuffs
    def onehot(self,inds,isSuffs=False):
        if(isSuffs):
            dupsuffs = [*inds]
            rawsuffs = []
            zerovect = np.zeros(self.rawsuffslen)
            # print(dupsuffs)
            for si in dupsuffs:
                if(si>-1):
                    zerovect[si] = 1
                rawsuffs.extend(zerovect.tolist())
                zerovect[si] = 0
            return rawsuffs
        else:
            dupprefs = [*inds]
            rawprefs = []
            zerovect = np.zeros(self.rawprefslen)
            # print(dupprefs)
            for pi in dupprefs:
                if(pi>-1):
                    zerovect[pi] = 1
                rawprefs.extend(zerovect.tolist())
                zerovect[pi] = 0
            return rawprefs
    def featuresOf(self,j,sent,word:str,prevtag:int,tagprobs:np.ndarray):
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
        return (np.array([
            len(sent),#length of sentence
            self.wordinds[sent[j]], #word
            1 if j==0 else 0,#is_first
            1 if j==len(sent)-1 else 0,#is_last
            iscapitalized,#iscapitalized
            isallcapitalized,#isallcapitalized,
            isalllower,
            # prevtag,#tag of word behind me
            self.wordinds[prevword],#prev_word
            self.wordinds[nextword],#next_word
            *self.rawprefindex(sent[j],3,False),#prefix-index
            *self.rawsuffindex(sent[j],3,False),#suffix-index
            # isnumeric,#isnumeric
            # -2 if j==0 else self.prefindex(sent[j-1]),
            # -2 if j==len(sent)-1 else self.suffindex(sent[j+1]),
            # *(tagprobs.tolist())
        ]))
    def getBatch(self,trainX:torch.Tensor,trainY:torch.Tensor):
        N = trainX.shape[0]
        perm = torch.randperm(N)
        trainX = trainX[perm]
        trainY = trainY[perm]
        for i in range(0,N,BATCHSZ):
            yield trainX[i:i+BATCHSZ],trainY[i:i+BATCHSZ] 
    def plot(self,val_accs, losses):
        """
            You can use this function to visualize progress of
            the training loss and validation accuracy  
        """
        plt.figure(figsize=(14,6))

        plt.subplot(1, 2, 1)
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Val Accuracy", fontsize=18)
        plt.plot(val_accs)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(1, 2, 2)
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Train Loss", fontsize=18)
        plt.plot(losses)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig('erracc.png')

    def trainNetwork(self,trainX:torch.Tensor,trainY:torch.Tensor):
        model = Network([trainX[0].shape[0],*hiddenLayers,NUMTAGS],LOSSFN=='CEL')
        losses = []
        accs = []
        opt = torch.optim.SGD(model.parameters(),lr=lr)
        for i in range(MAX_EPOCHS):
            model.train()
            totalloss = 0
            totalacc = 0
            print(f'Epoch: {i}/{MAX_EPOCHS}; Numiters: {int(trainX.shape[0]/BATCHSZ)}, #insts: {trainX.shape[0]}')
            for trainXSample,trainYSample in self.getBatch(trainX,trainY):
                # print(trainXSample.shape,trainYSample.shape)
                # continue
                opt.zero_grad()
                predVects = model(trainXSample)
                loss = model.loss(predVects,trainYSample)
                totalloss+=loss.item()
                loss.backward()
                opt.step()
            totalloss/=int(trainX.shape[0]/BATCHSZ)
            totalacc = model.accuracy(model(trainX),trainY)
            accs.append(totalacc)
            losses.append(totalloss)
            print(f'acc: {totalacc}; loss:{totalloss}')
        # self.plot(accs,losses)
        return model
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
        self.taginds.clear()
        for i,tag in enumerate(tagset):
            self.taginds[tag] = i
        mappedTags = []
        for tagsent in tags:
            mappedTags.append([])
            for i in range(len(tagsent)):
                mappedTags[-1].append(self.taginds[tagsent[i]])
        return mappedTags
    def mapWordsToFVsTrain(self,words,tags):
        allwords = []
        alltags = []
        preprocwords = self.preProcSents(words,True)
        for sent,tagsent in zip(preprocwords,tags):
            allwords.extend(sent)
            alltags.extend(tagsent)
        allwords.extend(['*',''])
        allwords = np.asarray(allwords)
        uniqwords = np.unique(allwords)
        self.wordinds.clear()
        for i in range(len(uniqwords)):
            self.wordinds[uniqwords[i]] = i
            
        featvects = []
        self.emmis = np.ones((len(uniqwords),NUMTAGS))
        for i in range(len(preprocwords)):
            for j in range(len(preprocwords[i])):
                self.emmis[self.wordinds[preprocwords[i][j]],tags[i][j]]+=1
        for i in range(len(preprocwords)):
            featvects.append([])
            for j in range(len(preprocwords[i])):
                prevtag = -1 if j==0 else tags[i][j-1]
                featvects[-1].append(self.featuresOf(j,preprocwords[i],words[i][j],prevtag,self.emmis[self.wordinds[preprocwords[i][j]]]))
        return featvects
    def saveSampleArrays(self, trainSents):
        words,tags = self.separateWordsTags(trainSents)
        taginds = self.mapTagsToInds(tags)
        featwords = self.mapWordsToFVsTrain(words,taginds)
        collapsed_featwords = []
        collapsed_taginds = []
        print("Mapped words and tags to numericals.")
        for i in range(len(featwords)):
            collapsed_featwords.extend(featwords[i])
            collapsed_taginds.extend(taginds[i])
        npfeats = np.array(collapsed_featwords)
        nptags = np.array(collapsed_taginds)
        np.savetxt('wordfeats.csv',npfeats)
        np.savetxt('tags.csv',nptags)
        self.rawprefslen = len(self.rawprefs)
        self.rawsuffslen = len(self.rawsuffs)
        np.savetxt('meta.csv',[self.rawprefslen,self.rawsuffslen],fmt='%d')
        # self.saveTagger(model)
        return npfeats,nptags
    def onehotify(self,npfeats:np.ndarray):
        feats = []
        for word in npfeats:
            word = word.tolist()
            feats.append([*word[:-6],*self.onehot(word[-6:-3]),*self.onehot(word[-3:],True)])
        return np.array(feats,dtype=np.float64)
    def trainOn(self,trainSents):
        try:
            npfeats = np.loadtxt('wordfeats.csv',dtype=np.int32)
            nptags = np.loadtxt('tags.csv')
            self.rawprefslen, self.rawsuffslen = np.loadtxt('meta.csv',dtype=int).tolist()
        except:
            print("Couldn't load npfeats. Generating from trainSents")
            npfeats,nptags = self.saveSampleArrays(trainSents)
        print("npfeats secured.")
        # npfeats = self.onehotify(npfeats)
        model = self.trainNetwork(torch.tensor(npfeats,dtype=torch.float),torch.tensor(nptags,dtype=torch.float))
        self.model = model
    def testOn(self, testSents):
        featwords = self.mapWordsToFVsTrain(testSents,False)
        answer = []
        for i in range(len(featwords)):
            tags = []
            answer.append([])
            tagprobs = self.model(torch.tensor(np.array(featwords[i])))
            for j,(wi,ti) in enumerate(zip((featwords[:,0]).numpy().tolist(),np.argmax(tagprobs,1))):
                w = testSents[i][j]
                t = self.tags[ti]
                answer[-1].append(f'{w}_{t}')
                tags.append(ti)
        return tags,answer
    def preProcSents(self,sents, train):
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                word = sents[i][j]
                word = word.lower()
                if(Tagger.isAmount(word)):
                    word = AMT
                elif(Tagger.isQualNum(word)):
                    word = QNW
                else:
                    prefset = set();suffset = set()
                    for k in range(min(MAXPSLENGTH,len(word))):
                        prefset.add(word[:(k+1)])
                        suffset.add(word[-(k+1):])
                    self.rawprefs = self.rawprefs.union(prefset)
                    self.rawsuffs = self.rawsuffs.union(suffset)
                if (train==False and (not self.wordinds.__contains__(word))):
                    word = '*'
                sents[i][j] = word
        self.rawprefs = list(self.rawprefs)
        self.rawsuffs = list(self.rawsuffs)
        self.rawprefinds = {k:v for v,k in enumerate(self.rawprefs)}
        self.rawsuffinds = {k:v for v,k in enumerate(self.rawsuffs)}
        self.rawprefinds['*'] = -1
        self.rawprefinds['*'] = -1
        self.rawprefinds[''] = -2
        self.rawsuffinds[''] = -2
        return sents
    def saveTagger(self,model:nn.Module):
        np.save("trained/emmis", self.emmis)
        np.save("trained/trellis", self.trellis)
        np.save("trained/init_prob", self.init_freqs)
        np.savetxt("trained/words.txt",self.words,fmt="%s")
        np.savetxt("trained/tags.txt",self.tags,fmt="%s")
    def loadTagger(self):
        self.emmis = np.load("trained/emmis.npy")
        self.trellis = np.load("trained/trellis.npy")
        self.init_freqs = np.load("trained/init_prob.npy")
        self.words = np.loadtxt("trained/words.txt",dtype='U')
        self.wordinds = {k:v for v,k in enumerate(self.words)}
        self.tags = np.loadtxt("trained/tags.txt",dtype='U')
        self.taginds = {k:v for v,k in enumerate(self.tags)}
        self.logemissionProbs = np.log10((self.emmis * (1 / np.tile(np.sum(self.emmis, 0), (len(self.words), 1)))).T)
        self.logtransitionProbs = np.log10(self.trellis * (1 / np.tile(np.sum(self.trellis, 1).reshape((len(self.tags), 1)), (1, len(self.tags)))))
        self.loginit_probs = np.log10(self.init_freqs/ np.sum(self.init_freqs))
    def demoSent(self,sent):
        sent2 = sent[:].split(' ')
        tags = self.testOn([sent.split(' ')])[0]
        for i in range(len(tags)):
            sent2[i]+= '_'+tags[i]
        return ' '.join(sent2)
    def findEvalMetrics(kfold):
        sents = list(brown.tagged_sents(tagset="universal"))
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
            POStagger.saveSampleArrays(trainSents)
            predTags = POStagger.testOn(testSentsOnlyWords)
            
            confmat, acc, pposa = POStagger.evalMetrics(predTags,testSentsOnlyTags)
            allitersppos.append(pposa)
            np.savetxt(f'results/pposa_{i+1}.csv',pposa,delimiter=', ')
            plt.imshow(confmat,cmap='hot',interpolation='nearest')
            plt.xticks(np.arange(12),labels=POStagger.tags,fontsize=8)
            plt.yticks(np.arange(12),labels=POStagger.tags,fontsize=8)
            plt.savefig(f'results/confmat_{i+1}.png')
            res.append(acc)
            print(f'Time for iteration {i+1}: {time() - now}')
            now = time()
        np.savetxt('results/perposmetrics.csv',np.mean(allitersppos,0))
        np.savetxt('results/accuracy.csv',res)
        return res
    def getTrainedTagger():
        t = Tagger()
        corpus = treebank
        dataset = list(corpus.tagged_sents(tagset="universal"))
        trr,tsr,vr = 0.6,0.2,0.2
        trainSents = dataset[:int(trr*len(dataset))]
        t.trainOn(trainSents)
        return t
if __name__=='__main__':
    t = Tagger.getTrainedTagger()