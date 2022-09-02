from typing import Tuple
import numpy as np
class Tagger:
    tags:list[str] = []#list of tags
    words:list[str] = []#list of words
    wordinds:dict#dictionary mapping words to indices in words.
    #helpful for mapping plural of a word to the same index if multiplexing nns and nn
    taginds:dict#dictionary mapping tags to indices in tags.
    trellis:list[list[float]]#transition probabilities size = len(tags)+2 (for . and ^)
    emmis:list[list[float]]#Emission probabilities. emmis[i][j] = P(word[i]|tag i in tags field)
    #-1=>'^'
    #-2=>'.'
    def dummyWordsTags(self)->list[list[str,int]]:
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
    def tokenizeTextToSentences(text:str)->list[list[str]]:
        """Converts supplied text into tokenized sentences.
        ### Also updates tags field in Tagger class with new tags as encountered.
        ### Invariant: tags = list of unique tags encountered so far.

        Args:
            text (str): Raw text read in from a file in brown/ folder

        Returns:
            list[list[list[str][2]]]: List of sentences: each sentence is a list of words: each word is list of two strings: word string and tag string.
            
        """
        pass
    def tagStringToIndex(sentences: list[list[list[str]]])->list[list[list[str,int]]]:
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
        self.emmis[-1,-1] = 1
        self.emmis[-2,-2] = 1
    def updateTrellisAndEmmis(self,sentences:list[list[list[str,int]]])->None:
        """Iterates through sentences (with optimizations), filling in trellis and emmis.

        Args:
            sentences (list[list[list[str,int]]]): _description_
        """
        tmpemmis = np.zeros_like(self.emmis)
        tmptrellis = np.zeros_like(self.trellis)
        for sent in sentences:
            pwi,pti = sent[0]
            tmpemmis[pwi,pti]+=1
            tmptrellis[-1,pti]+=1 # P(ti | ^)
            for wi,ti in sent[1:]:
                tmptrellis[pti,ti]+=1
                tmpemmis[wi,ti]+=1
                pti = ti
            tmptrellis[pti,-2]+=1 #P(.|ti)
        self.emmis += tmpemmis
        self.trellis += tmptrellis
    def transmissionProbability(self,pti,ti)->float:
        """returns transmission probability

        Args:
            pti (_type_): tag at i-1
            ti (_type_): tag at i

        Returns:
            float: P(ti|ti-1)
        """
        return self.trellis[pti,ti]/np.sum(self.trellis[pti,:])
    def emmissionProbability(self,ti,wi)->float:
        """returns P(wi|ti)

        Args:
            ti (_type_): tag at position i
            wi (_type_): word at position i

        """
        return self.emmis[wi,ti]/np.sum(self.emmis[:,ti])
    
    def findTagSequence(self,sentence:str)->str:
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
if __name__=='__main__':        
    tagger = Tagger()
    sents = tagger.dummyWordsTags()
    for sent in sents:
        print(list(map(lambda wt:tagger.words[wt[0]],sent)))
    tagger.initializeTrellisAndEmmis()
    tagger.updateTrellisAndEmmis(sents)

    # ###Test:
    testSent = "the fox hunted for the man"
    tagger.findTagSequence(testSent)
