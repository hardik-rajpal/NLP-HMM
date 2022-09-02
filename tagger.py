from typing import Tuple
import numpy as np
import nltk
from nltk.corpus import brown
class Tagger:
    tags = []#list of tags
    words = []#list of words
    wordinds = {}#dictionary mapping words to indices in words.
    #helpful for mapping plural of a word to the same index if multiplexing nns and nn
    taginds = {}#dictionary mapping tags to indices in tags.
    init_prob = []
    train_size = 2000
    # trellis#transition probabilities size = len(tags)+2 (for . and ^)
    # emmis#Emission probabilities. emmis[i][j] = P(word[i]|tag i in tags field)
    #-1=>'^'
    #-2=>'.'
    def dummyWordsTags(self):
        """

        Returns:
            list[list[str,int]]:  a list of dummy sentences after initializing unique tags and words.
        """
        self.tags = ['N','V','A','DT','Adv','P','.','^']
        self.words = ['the','fox','dances','weirdly','man','hunted','for','.','^']
        self.taginds = {
            'N':0,
            'V':1,
            'A':2,
            'DT':3,
            'Adv':4,
            'P':5,
            '.':-2,
            '^':-1
        }
        self.wordinds = {
            'the':0,
            'fox':1,
            'dances':2,
            'weirdly':3,
            'man':4,
            'hunted':5,
            'for':6,
            '.':-2,
            '^':-1
        }
        return [
            [
                [0,3],[1,0],[2,1],[3,4]
            ],
            [
                [0,3],[4,0],[5,1],[6,5],[0,3],[1,0]
            ]
        ]
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
        
        self.trellis = np.zeros((len(self.tags),len(self.tags)))
        self.emmis = np.zeros((len(self.words),len(self.tags)))
        self.init_prob = np.zeros((len(self.tags)))

    def updateTrellisAndEmmis(self,sentences):
        """Iterates through sentences (with optimizations), filling in trellis and emmis.

        Args:
            sentences (list[list[list[str,int]]]): _description_
        """
        tmpemmis = np.zeros_like(self.emmis)
        tmptrellis = np.zeros_like(self.trellis)

        for sent in sentences:
            pwi, pti = sent[0]
            tmpemmis[pwi,pti] += 1
            
            self.init_prob[pti] += 1
            
            for wi,ti in sent[1:]:
                tmptrellis[pti,ti] += 1
                tmpemmis[wi,ti] += 1
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
        numtags = len(self.tags)
        confmat = np.zeros((numtags,numtags))
        total = len(preds)

        for pred,trueTag in zip(preds,trueTags):
            confmat[pred,trueTag]+=1
        accuracy = np.sum(np.array([confmat[i,i] for i in range(numtags)]))/total
        perPOS_acc = [confmat[i,i]/(max(1,np.sum(confmat[:,i]))) for i in np.arange(numtags)]
        return confmat,accuracy,perPOS_acc
    
    def findTagSequence(self,sentence):
        """Should return the given sentence with pos tags attached to each word/punctuation.

        Args:
            sentence (str): input sentence

        Returns:
            str: output sentence with pos tags attached to each word.
        """
        words = sentence.split(' ')
        words = ['^',*words,'.']
        T = len(words)
        N = self.emmis.shape[0]
        K = self.trellis.shape[0]
        T1 = np.zeros((K,T))
        T2 = np.zeros((K,T))
        for i in range(K):
            T1[i,1] = 1/K*1
            T2[i,1] = 0
        for j in range(1,T):
            for i in range(K):
                T1[i,j] = self.emmissionProbability(i,self.wordinds[words[j]])*np.max(np.vectorize(lambda k:T1[k,j-1]*self.transmissionProbability(k,j))(np.arange(K)))
                T2[i,j] = np.argmax(np.vectorize(lambda k:T1[k,j-1]*self.transmissionProbability(k,i))(np.arange(K)))*self.emmissionProbability(i,self.wordinds[words[j]])
    def trainOn(self, trainSents):
        self.initializeTrellisAndEmmis()
        self.updateTrellisAndEmmis(trainSents)
    def testOn(self, testSents):
        
    def sentencesFromCorpus(self):

        train_size = self.train_size
        corpus = brown.tagged_sents()[:train_size]
        sentences = [sent for sent in corpus]
        
        tagged_words = []
        for sent in sentences:
            for word in sent:
                tagged_words.append(word)
        tagged_words = np.asarray(tagged_words)
        
        tags = np.unique(tagged_words[:,1])
        
        words_all = tagged_words[:, 0]
        words_all = list(map(lambda x: x.lower(), words_all))
        # print(f'total words: {len(words_all)}')
        words,counts = np.unique(words_all,return_counts=True)
        np.savetxt('words.csv',words,fmt="%s")
        np.savetxt('counts.csv',counts,fmt="%d")
        
        self.tags = tags
        self.words = words
        # print(f'words ({len(self.words)}): {self.words} \n tags ({len(self.tags)}): {tags}')
        for i in range(len(self.words)):
            self.wordinds[self.words[i]] = i
        
        for i in range(len(self.tags)):
            self.taginds[self.tags[i]] = i
        
        mapped_sentences = [[ [self.wordinds[a.lower()], self.taginds[b]] for a, b in sent] for sent in sentences]
        print(sentences[0], mapped_sentences[0])

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
    
    def viterbi(self, Y):
        
        K = len(self.tags)
        N = len(self.words)
        N_curr = len(Y)

        pi = self.init_prob.reshape((K))

        A = self.trellis * (1 / np.tile(np.sum(self.trellis, 1).reshape((K, 1)), (1, len(self.tags))))
        A = A.reshape((K,K))

        B = self.emmis * (1 / np.tile(np.sum(self.emmis, 0), (len(self.words), 1)))
        B = B.T
        B = B.reshape((K,N))
        
        S = self.tags.reshape((K,1))
        
        Y = np.asarray(Y).reshape((N_curr))
        
        # for i in range(len(self.words)):
        #     for j in range(len(self.tags)):
        #         B[i,j] = self.emmis[i,j]/np.sum(self.emmis[:,j])

        T1 = np.zeros((K, N_curr))
        T2 = np.zeros((K, N_curr))

        T1[:, 0] = pi * B[:, int(Y[0])]

        for obs in range(1, N_curr):

            # B_factor[0, i] represents the probabilty P(word_i | state)
            B_factor = B[:, int(Y[obs])].reshape(1,K)

            # B_matrix_factor is row wise repetiton of B_factor
            B_matrix_factor = np.tile(B_factor, (K,1))
        
            # A_factor = A P( s_j | s_i) * P(word | s_j)
            A_factor = A * B_matrix_factor

            temp = np.tile(T1[:, obs-1].reshape(-1, 1), (1, K))
            # final = P( prev ending s_i) * P(s_j | s_i) * P(word| s_j)
            final = temp * A_factor
        
            T1[:, obs] = np.max(final, 1)
            T2[:, obs] = np.argmax(final, 1)
        
        point = np.argmax(T1[:, N_curr-1])
        best_path = []

        for obs in range(N_curr-2, -1, -1):
            best_path.append(S[point].item())
            point = int(T2[point, obs])

        return best_path

tagger = Tagger()
sents = tagger.sentencesFromCorpus()
# for sent in sents:
#     print(list(map(lambda wt:tagger.words[wt[0]],sent)))
tagger.initializeTrellisAndEmmis()
tagger.updateTrellisAndEmmis(sents)
# tagger.saveTagger()
# tagger.loadTagger()
# sent = tagger.viterbi([6667, 2959, 1884, 3131, 3766])
# print(sent)
exit()



# ###Test:
testSent = "the fox hunted for the man"
tagger.findTagSequence(testSent)
