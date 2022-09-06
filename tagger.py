from time import time
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import brown
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
    def tokenizeTextToSentences(text):
        """Converts supplied text into tokenized sentences.
        ### Also updates tags field in Tagger class with new tags as encountered.
        ### Invariant: tags = list of unique tags encountered so far.

        Args:
            text (str): Raw text read in from a file in brown/ folder

        Returns:
            list[list[list[str][2]]]: List of sentences: each sentence is a list of words: each word is list of two strings: word string and tag string.
            
        """
        pass
    def tagStringToIndex(sentences):
        """Converts tag strings to indices in tags field. Should be used after collecting all possible tags into tags field.

        Args:
            sentences (list[list[list[str,str]]]): 

        Returns:
            list[list[list[str,int]]]: 
        """
        pass
    def initializeTrellisAndEmmis(self):
        """Initializes Trellis and Emmis to arrays of appropriate dimensions.
        """
        
        self.trellis = np.ones((len(self.tags),len(self.tags)))
        self.emmis = np.ones((len(self.words),len(self.tags)))
        self.init_freqs = np.ones((len(self.tags)))
    def updateTrellisAndEmmis(self,sentences,lfws):
        """Iterates through sentences (with optimizations), filling in trellis and emmis.

        Args:
            sentences (list[list[list[str,int]]]): _description_
        """
        tmpemmis = np.zeros_like(self.emmis)
        tmptrellis = np.zeros_like(self.trellis)

        for sent in sentences:
            pwi, pti = sent[0]
            tmpemmis[pwi,pti] += 1
            if(self.freqMap[self.words[pwi]]<=2):
                tmpemmis[self.wordinds['*'],pti] += 1
            self.init_freqs[pti] += 1
            
            for wi,ti in sent[1:]:
                tmptrellis[pti,ti] += 1
                tmpemmis[wi,ti] += 1
                if(self.freqMap[self.words[wi]]<=2):
                    tmpemmis[self.wordinds['*'],ti] += 1
                pti = ti
        
        self.emmis += tmpemmis
        self.trellis += tmptrellis
        self.init_freqs += self.init_freqs
        self.logemissionProbs = np.log10((self.emmis * (1 / np.tile(np.sum(self.emmis, 0), (len(self.words), 1)))).T)
        self.logtransitionProbs = np.log10(self.trellis * (1 / np.tile(np.sum(self.trellis, 1).reshape((len(self.tags), 1)), (1, len(self.tags)))))
        self.loginit_probs = np.log10(self.init_freqs/ np.sum(self.init_freqs))
    def transmissionProbability(self,pti,ti):
        """returns transmission probability

        Args:
            pti (_type_): tag at i-1
            ti (_type_): tag at i

        Returns:
            float: P(ti|ti-1)
        """
        return self.trellis[pti,ti]/np.sum(self.trellis[pti,:])
    def emmissionProbability(self,ti,wi):
        """returns P(wi|ti)

        Args:
            ti (_type_): tag at position i
            wi (_type_): word at position i

        """
        return self.emmis[wi,ti]/np.sum(self.emmis[:,ti])    
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
    def trainOn(self, trainSents):
        mapped_sents,leastFreqWords = self.get_mapped_sentences(trainSents)
        self.initializeTrellisAndEmmis()
        self.updateTrellisAndEmmis(mapped_sents,leastFreqWords)
        self.saveTagger()
    def testOn(self, testSents):
        mapped_sents = self.get_mapped_sentences_test(testSents)
        answer = []
        for sent in mapped_sents:
            best_path = self.viterbi(sent)
            answer.append(best_path)
        return answer

    def preProcSents(self,sents, train=True):
        if train==True:
            # qualtags = []
            for i in range(len(sents)):
                for j in range(len(sents[i])):
                    word = list(sents[i][j])
                    word[0] = word[0].lower()
                    if(Tagger.isAmount(word[0])):
                        word[0] = '$amt$'
                    if(Tagger.isQualNum(word[0])):
                        word[0] = f'$qnw$'
                    sents[i][j] = word
        else:
            for i in range(len(sents)):
                for j in range(len(sents[i])):
                    word = sents[i][j].lower()
                    if(Tagger.isAmount(word)):
                        word = '$amt$'
                    if(Tagger.isQualNum(word)):
                        word = f'$qnw$'
                    if (not self.wordinds.__contains__(word)):
                        word = '*'
                    sents[i][j] = word
        return sents
    def get_mapped_sentences(self, sents):
        sentences = sents        
        tagged_words = []
        sentences = self.preProcSents(sentences)
        for sent in sentences:
            for word in sent:
                tagged_words.append(word)
        tagged_words = np.asarray(tagged_words)
        tags = np.unique(tagged_words[:,1])
        
        words_all = tagged_words[:, 0]
        words_all = list(map(lambda x: x, words_all))
        words,counts = np.unique(words_all,return_counts=True)
        np.savetxt('words.csv',words,fmt="%s")
        freqThres = 2
        for i in range(len(counts)):
            self.freqMap[words[i]] = counts[i]
        leastFreqWords = words[counts<=freqThres].tolist()
        leastFreqWords = []
        self.tags = tags
        self.words = words.tolist()
        self.words.append('*')
        self.words = np.asarray(self.words)
        self.wordinds.clear()
        for i in range(len(self.words)):
            self.wordinds[self.words[i]] = i
        self.taginds.clear()
        for i in range(len(self.tags)):
            self.taginds[self.tags[i]] = i
        
        mapped_sentences = [[ [self.wordinds[a], self.taginds[b]] for a, b in sent] for sent in sentences]

        return mapped_sentences,leastFreqWords
    def get_mapped_sentences_test(self, sents):
        corpus = sents
        sentences = [sent for sent in corpus]
        sentences = self.preProcSents(sentences, False)
        mapped_sentences = [[ self.wordinds[a] for a in sent] for sent in sentences]
        return mapped_sentences

    def saveTagger(self):
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
    def viterbi(self, wordlist):#(observations, states, start_p, trans_p, emit_p)
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
def findEvalMetrics(k):
    sents = list(brown.tagged_sents(tagset="universal"))
    l = len(sents)
    perm = np.random.permutation(l)
    sents = np.asanyarray(sents,dtype=object)[perm].tolist()
    res = []
    now = time()
    allitersppos = []
    for i in range(k):
        p1 = sents[:int((i*l)/k)]
        p2 = sents[int((i+1)*l/k):]
        dummyTest = sents[int((-1*l)/k):]
        p1.extend(p2)
        trainSents = p1
        testSents = sents[int(i*l/k):int((i+1)*l/k)]
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
        POStagger.saveTagger()
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
def getTrainedModel():
    sents = list(brown.tagged_sents(tagset="universal"))
    l = len(sents)
    perm = np.random.permutation(l)
    sents = np.asanyarray(sents,dtype=object)[perm].tolist()
    tagger = Tagger()
    tagger.trainOn(sents[:int(4*l/5)])
    return tagger
if __name__=='__main__':
    findEvalMetrics(5)

    # tagger.saveTagger()