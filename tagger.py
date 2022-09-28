from time import time
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import brown
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
layerszs = [12]
MAX_EPOCHS = 50
BATCHSZ = 256
lr = 0.01
 
class Network(nn.Module):
    def __init__(self, lyrszs:list[int]):
        #lyrszs[0] = 12 (#features)
        #lyrszs[-1] = 12 (#tags)
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1,len(lyrszs)):
            self.layers.append(nn.Linear(lyrszs[i-1],lyrszs[i]))
            self.layers.append(nn.Softmax(1))
    def forward(self,X:list[int]):
        for layer in self.layers:
            X = layer(X)
        return X
    def loss(self,predvects,labels:torch.Tensor):
        labvects = torch.zeros_like(predvects)
        for x in list(zip(torch.tensor(np.arange(len(labels),dtype=int),dtype=torch.long),labels.type(torch.long))):
            labvects[x[0],x[1]] = 1
        losses = (labvects-predvects)**2
        return torch.mean(torch.sum(losses,1))
    def accuracy(self,predvects,labels):
        return np.round(100*(labels[torch.argmax(predvects,1)==labels].shape[0]/labels.shape[0]),2)
class Tagger:
    NUMTAGS = 12
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
        for i,pref in enumerate(self.prefixes):
            self.prefinds[pref] = i
        for i,suff in enumerate(self.suffixes):
            self.suffinds[suff] = i
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
    def prefindex(self,word:str):
        for i in range(1,len(word)):
            if(self.prefinds.__contains__(word[:i])):
                return self.prefinds[word[:i]]
        return 0
    def suffindex(self,word):
        for i in range(1,len(word)):
            if(self.suffinds.__contains__(word[-i:])):
                return self.suffinds[word[-i:]]
        return 0
    def featuresOf(self,j,sent,word:str):
        prevword = '' if j==0 else sent[j-1]
        nextword = '' if j+1==len(sent) else sent[j+1]
        isnumeric = 1 if sent[j] == AMT or sent[j] == QNW else 0
        iscapitalized = 0
        isallcapitalized = 0
        if isnumeric == 0:
            if ((word.lower()!=sent[j])):
                iscapitalized = 1
            elif(word.upper()==word):
                isallcapitalized = 1
        return (np.array([
            self.wordinds[sent[j]], #word
            1 if j==0 else 0,#is_first
            1 if j==len(sent)-1 else 0,#is_last
            self.wordinds[prevword],#prev_word
            self.wordinds[nextword],#next_word
            0 if isnumeric else self.prefindex(sent[j]),#prefix-index
            0 if isnumeric else self.suffindex(sent[j]),#suffix-index
            isnumeric,#isnumeric
            iscapitalized,#iscapitalized
            isallcapitalized,#isallcapitalized
        ]))
    def getBatch(self,trainX:torch.Tensor,trainY:torch.Tensor):
        N = trainX.shape[0]
        perm = torch.randperm(N)
        trainX = trainX[perm]
        trainY = trainY[perm]
        for i in range(BATCHSZ,N,BATCHSZ):
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
        plt.show()

    def trainNetwork(self,trainX:torch.Tensor,trainY:torch.Tensor):
        model = Network([trainX[0].shape[0],*layerszs,Tagger.NUMTAGS])
        losses = []
        accs = []
        opt = torch.optim.SGD(model.parameters(),lr=lr)
        for i in range(MAX_EPOCHS):
            model.train()
            totalloss = 0
            totalacc = 0
            print(f'Numiters: {int(trainX.shape[0]/BATCHSZ)}, #insts: {trainX.shape[0]}')
            for trainXSample,trainYSample in self.getBatch(trainX,trainY):
                # print(trainXSample.shape,trainYSample.shape)
                # continue
                opt.zero_grad()
                predVects = model(trainXSample)
                loss = model.loss(predVects,trainYSample)
                totalloss+=loss.item()
                totalacc+=(model.accuracy(predVects,trainYSample))
                loss.backward()
                opt.step()
            totalloss/=int(trainX.shape[0]/BATCHSZ)
            totalacc/=int(trainX.shape[0]/BATCHSZ)
            accs.append(totalacc)
            losses.append(totalloss)
            print(f'acc: {totalacc}; loss:{totalloss}')
        self.plot(accs,losses)
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
            if(len(tagset)==Tagger.NUMTAGS):
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
    def mapWordsToFVs(self,words,train=True):
        allwords = []
        preprocwords = self.preProcSents(words,train)
        for sent in preprocwords:
            allwords.extend(sent)
        allwords.extend(['*',''])
        allwords = np.asarray(allwords)
        uniqwords = np.unique(allwords)
        self.wordinds.clear()
        for i in range(len(uniqwords)):
            self.wordinds[uniqwords[i]] = i
        featvects = []
        for i in range(len(preprocwords)):
            featvects.append([])
            for j in range(len(preprocwords[i])):
                featvects[-1].append(self.featuresOf(j,preprocwords[i],words[i][j]))
        return featvects
    def trainOn(self, trainSents):
        words,tags = self.separateWordsTags(trainSents)
        featwords = self.mapWordsToFVs(words)
        taginds = self.mapTagsToInds(tags)
        collapsed_featwords = []
        collapsed_taginds = []
        for i in range(len(featwords)):
            collapsed_featwords.extend(featwords[i])
            collapsed_taginds.extend(taginds[i])
        model = self.trainNetwork(torch.tensor(np.array(collapsed_featwords,dtype=np.float32)),torch.tensor(np.array(collapsed_taginds,dtype=np.float32)))
        self.model = model
        # self.saveTagger(model)
    def testOn(self, testSents):
        featwords = self.mapWordsToFVs(testSents,False)
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
                if(Tagger.isQualNum(word)):
                    word = QNW
                if (train==False and (not self.wordinds.__contains__(word))):
                    word = '*'
                sents[i][j] = word
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
    def viterbi(self, wordlist):
        states = np.arange(len(self.tags))
        wordlist = np.asarray(wordlist)
        logemissionProbs = np.log10((self.emmis * (1 / np.tile(np.sum(self.emmis, 0), (len(self.words), 1)))).T)
        logtransitionProbs = np.log10(self.trellis * (1 / np.tile(np.sum(self.trellis, 1).reshape((len(self.tags), 1)), (1, len(self.tags)))))
        loginit_probs = np.log10(self.init_freqs/ np.sum(self.init_freqs))
        V = np.zeros((len(wordlist),len(states),2))
        V[0] = np.tile(
            (
                (np.array([loginit_probs.flatten()+(logemissionProbs[:,wordlist[0]]).flatten()]))
            ).T,
            reps=2)
        for t in range(1, len(wordlist)):
            constvt1 = np.tile(np.array([V[t-1][:,0].flatten()]).T,len(states))
            vals = constvt1+logtransitionProbs # 12x12
            stateSel = np.argmax(vals,0).flatten()
            transProbMax = np.max(vals,0).flatten()+logemissionProbs[:,wordlist[t]].flatten()
            V[t] = np.stack([transProbMax,stateSel],1)
        maxProb = -1000
        previousState = None
        outputs = []
        maxProbInd = np.argmax(V[-1,:,0])
        previousState = int(V[-1,:,1][maxProbInd])
        maxProbVal = V[-1,:,0][maxProbInd]
        if (maxProbVal<maxProb):
            print("Probability is 0")
            return []
        outputs.append(maxProbInd)
        for t in range(len(V) - 2, -1, -1):
            outputs.insert(0, V[t + 1][previousState][1])
            previousState = int(V[t + 1][previousState][1])
        for i in range(len(outputs)):
            outputs[i] = self.tags[int(outputs[i])]
        # print(' '.join(list(map(lambda w,t: f'{w}_{t}',list(map(lambda x:(self.words[x]),Y.tolist())),outputs))))
        # exit()
        return outputs
    def demoSent(self,sent):
        sent2 = sent[:].split(' ')
        tags = self.testOn([sent.split(' ')])[0]
        for i in range(len(tags)):
            sent2[i]+= '_'+tags[i]
        return ' '.join(sent2)
    def getTrainedModel():
        sents = list(brown.tagged_sents(tagset="universal"))
        l = len(sents)
        perm = np.random.permutation(l)
        sents = np.asanyarray(sents,dtype=object)[perm].tolist()
        tagger = Tagger()
        tagger.trainOn(sents[:int(4*l/5)])
        return tagger
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
            POStagger.trainOn(trainSents)
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
if __name__=='__main__':
    t = Tagger()
    sents = list(brown.tagged_sents(tagset="universal"))
    k = 5
    sents = sents[:int(len(sents)/k)]
    t.trainOn(sents)
