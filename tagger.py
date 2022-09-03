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
        self.init_prob = []
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
        self.init_prob = np.ones((len(self.tags)))
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
            self.init_prob[pti] += 1
            
            for wi,ti in sent[1:]:
                tmptrellis[pti,ti] += 1
                tmpemmis[wi,ti] += 1
                if(self.freqMap[self.words[wi]]<=2):
                    tmpemmis[self.wordinds['*'],ti] += 1
                pti = ti
        
        self.emmis += tmpemmis
        self.trellis += tmptrellis

        #TODO : modify below to accomodate sentences in fly
        self.init_prob = self.init_prob / len(sentences)
        np.savetxt('emmis.csv',self.emmis)
        np.savetxt('trellis.csv',self.trellis)
        np.savetxt('initprob.csv',self.init_prob)
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
        # print(preds[:10])
        # print(trueTags[:10])
        # exit()
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
        perPOS_acc = [confmat[i,i]/(max(1,np.sum(confmat[:,i]))) for i in np.arange(numtags)]
        return confmat,accuracy,perPOS_acc
    def trainOn(self, trainSents):
        mapped_sents,leastFreqWords = self.get_mapped_sentences(trainSents)
        
        self.initializeTrellisAndEmmis()
        self.updateTrellisAndEmmis(mapped_sents,leastFreqWords)
        np.savetxt('words.txt',self.words,fmt="%s")
        np.savetxt('tags.txt',self.tags,fmt="%s")
        np.savetxt('trellis.csv',self.trellis,fmt="%d")
        np.savetxt('emmis.csv',self.emmis,fmt='%d')  
    def testOn(self, testSents):
        mapped_sents = self.get_mapped_sentences_test(testSents)
        answer = []
        for sent in mapped_sents:
            best_path = self.viterbi(sent)
            answer.append(best_path)
            # print(answer)
            # exit()
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
                        # qualtags.append(word[1])
                    sents[i][j] = word
            # qt,c = np.unique(np.array(qualtags),return_counts=True)
            # np.savetxt('qualtags.csv',qt,fmt="%s")
            # np.savetxt('counttags.csv',c,fmt="%d")
        else:
            # qualtags = []
            for i in range(len(sents)):
                for j in range(len(sents[i])):
                    word = sents[i][j].lower()
                    if(Tagger.isAmount(word)):
                        word = '$amt$'
                    if(Tagger.isQualNum(word)):
                        word = f'$qnw$'
                        # qualtags.append(word[1])
                    if (not self.wordinds.__contains__(word)):
                        word = '*'
                    sents[i][j] = word
            # qt,c = np.unique(np.array(qualtags),return_counts=True)
            # np.savetxt('qualtags.csv',qt,fmt="%s")
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
        # print(f'total words: {len(words_all)}')
        words,counts = np.unique(words_all,return_counts=True)
        # np.savetxt('words.csv',words,fmt="%s")
        freqThres = 2
        for i in range(len(counts)):
            self.freqMap[words[i]] = counts[i]
        leastFreqWords = words[counts<=freqThres].tolist()
        leastFreqWords = []
        # np.savetxt('counts.csv',counts,fmt="%d")
        
        self.tags = tags
        self.words = words.tolist()
        self.words.append('*')
        self.words = np.asarray(self.words)
        # print(f'words ({len(self.words)}): {self.words} \n tags ({len(self.tags)}): {tags}')
        self.wordinds.clear()
        for i in range(len(self.words)):
            self.wordinds[self.words[i]] = i
        self.taginds.clear()
        for i in range(len(self.tags)):
            self.taginds[self.tags[i]] = i
        
        mapped_sentences = [[ [self.wordinds[a], self.taginds[b]] for a, b in sent] for sent in sentences]
        # print(sentences[0], mapped_sentences[0])

        return mapped_sentences,leastFreqWords
    def get_mapped_sentences_test(self, sents):
        corpus = sents
        sentences = [sent for sent in corpus]
        
        sentences = self.preProcSents(sentences, False)
        #TODO: smoothing function
        mapped_sentences = [[ self.wordinds[a] for a in sent] for sent in sentences]
        # print(sentences[0], mapped_sentences[0])

        return mapped_sentences

    def saveTagger(self):
        np.save("emmis", self.emmis)
        np.save("trellis", self.trellis)
        np.save("init_prob", self.init_prob)
    def loadTagger(self):
        #TODO: rewrite this part
        tagged_words = np.asarray(brown.tagged_words())
        self.tags = np.unique(tagged_words[:,1])
        self.words = np.unique(tagged_words[:,0])
        self.emmis = np.load("emmis.npy")
        self.trellis = np.load("trellis.npy")
        self.init_prob = np.load("init_prob.npy")
    def dptable(self, V):
        
        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
    def viterbi(self, Y):#(observations, states, start_p, trans_p, emit_p)

        K = len(self.tags)
        states = np.arange(len(self.tags))
        observations = np.asarray(Y)

        trans_p = self.trellis * (1 / np.tile(np.sum(self.trellis, 1).reshape((K, 1)), (1, len(self.tags))))

        B = self.emmis * (1 / np.tile(np.sum(self.emmis, 0), (len(self.words), 1)))

        emit_p = B.T
        pi = self.init_prob

        V = [{}]
        for st in states:
            V[0][st] = {"prob": pi[st] * emit_p[st][observations[0]], "prev": None}
   
        for t in range(1, len(observations)):
            V.append({})
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
    
                max_prob = max_tr_prob * emit_p[st][observations[t]]
                V[t][st] = {"prob": max_prob*100, "prev": prev_st_selected}
    
        opt = []
        max_prob = 0.0
        best_st = None
    
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        if previous is None:
            # TODO: handle this case:
            print("Probability is 0")
            return []
    
        for t in range(len(V) - 2, -1, -1):
            # print(previous)
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        # print ("The steps of states are " + " ".join(self.tags[opt]) + " with highest probability of %s" % max_prob)
        for i in range(len(opt)):
            opt[i] = self.tags[opt[i]]
        
        return opt
def findEvalMetrics(k):
    sents = list(brown.tagged_sents(tagset="universal"))
    l = len(sents)
    perm = np.random.permutation(l)
    sents = np.asanyarray(sents,dtype=object)[perm].tolist()
    # print(sents[0])
    res = []
    now = time()
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
        # print(testSentsOnlyTags[:10])
        POStagger = Tagger()
        POStagger.trainOn(trainSents)
        predTags = POStagger.testOn(testSentsOnlyWords)
        
        confmat, acc, pposa = POStagger.evalMetrics(predTags,testSentsOnlyTags)
        np.savetxt(f'confmat_{i+1}.csv',confmat,delimiter=', ')
        np.savetxt(f'pposa_{i+1}.csv',pposa,delimiter=', ')
        plt.imshow(confmat,cmap='hot',interpolation='nearest')
        plt.savefig(f'confmat_{i+1}.png')
        res.append(acc)
        print(f'Time for iteration: {i+1}: {time() - now}')
        now = time()
        #TODO remove break below 
        # break
    np.savetxt('accuracy.csv',res)
    return res
if __name__=='__main__':
    # tagger = Tagger()
    # sents = list(brown.tagged_sents(tagset='universal'))
    # train = sents[:2000]
    # test = sents[2000:2002]
    # testSents = list(map(
    #                 lambda sent:list(map(lambda wt:wt[0],sent)),
    #                 test
    #                 ))
    # testTags = list(map(
    #     lambda sent:list(map(lambda wt:wt[1],sent)),
    #     test
    # ))
    # tagger.trainOn(train)
    # preds = tagger.testOn(testSents)
    # evals = []
    # for i in range(len(preds)):
    #     # print()
    #     evals.append(list(zip(testSents[i],preds[i],test[i])))
    # np.savetxt('evals.txt',np.asarray(evals),fmt="%s")
    # # for sent in sents:
    # #     print(list(map(lambda wt:tagger.words[wt[0]],sent)))
    # # tagger.saveTagger()
    # # tagger.loadTagger()
    # # sent = tagger.viterbi([6667, 2959, 1884, 3131, 3766])
    # # print(sent)
    # exit()
    findEvalMetrics(5)


    # ###Test:
    # testSent = "the fox hunted for the man"
