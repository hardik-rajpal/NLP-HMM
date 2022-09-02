from typing import List, Tuple
import tagger
import numpy as np
from nltk.corpus import brown
def findEvalMetrics(k):
    sents = list(brown.tagged_sents())
    l = len(sents)
    perm = np.random.permutation(l)
    sents = np.asanyarray(sents,dtype=object)[perm].tolist()
    res = []
    for i in range(k):
        p1 = sents[:int((i*l)/k)]
        p2 = sents[int((i+1)*l/k):]
        p1.extend(p2)
        trainSents = p1
        testSents = sents[int(i*l/k):int((i+1)*l/k)]
        testSents = list(map(
                        lambda sent:list(map(lambda wt:wt[0],sent)),
                        testSents
                        ))
        testTags = list(map(
            lambda sent:list(map(lambda wt:wt[1],sent)),
            testSents
        ))
        POStagger = tagger.Tagger()
        POStagger.trainOn(trainSents)
        predTags = POStagger.testOn(testSents)
        confmat, acc,pposa = POStagger.evalMetrics(predTags,testTags)
        np.savetxt(f'confmat_{i+1}.csv',confmat)
        np.savetxt(f'pposa_{i+1}.csv',confmat)
        res.append(acc)
    np.savetxt('accuracy.csv',res)
    return res
if __name__=='__main__':
    findEvalMetrics(5)