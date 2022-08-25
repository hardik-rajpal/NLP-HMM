from typing import Tuple
class Tagger:
    tags:list[str] = []#list of tags
    words:list[str] = []#list of words
    trellis:list[list[float]]#transition probabilities size = len(tags)+2 (for . and ^)
    emmis:list[dict]#Emission probabilities. emmis[i]['people'] = P('people'|tag i in tags field)
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
    def tagStringToIndex(sentences: list[list[list[str][2]]])->list[list[list[str,int]]]:
        """Converts tag strings to indices in tags field. Should be used after collecting all possible tags into tags field.

        Args:
            sentences (list[list[list[str,str]]]): 

        Returns:
            list[list[list[str,int]]]: 
        """
        pass
    def initializeTrellisAndEmmis():
        """Initializes Trellis and Emmis to arrays of appropriate dimensions.
        """
        pass
    def updateTrellisAndEmmis(sentences:list[list[list[str,int]]])->None:
        """Iterates through sentences (with optimizations), filling in trellis and emmis.

        Args:
            sentences (list[list[list[str,int]]]): _description_
        """
        pass
    def findTagSequence(sentence:str)->str:
        """Should return the given sentence with pos tags attached to each word/punctuation.

        Args:
            sentence (str): input sentence

        Returns:
            str: output sentence with pos tags attached to each word.
        """
        pass

tagger = Tagger()
sents = tagger.tokenizeTextToSentences(open('brown/ca01',mode='r').read())
sents = tagger.tagStringToIndex(sents)
tagger.initializeTrellisAndEmmis()
tagger.updateTrellisAndEmmis(sents)
###Test:
testSent = "The quick brown fox jumped over."
tagger.findTagSequence(testSent)
