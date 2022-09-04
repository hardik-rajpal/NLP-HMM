import numpy as np
scores = np.loadtxt('perposmetrics.csv',delimiter=', ')
emmis = np.loadtxt('emmis.csv')
prec = scores[0,:]
rec = scores[1,:]
beta = 0.5
fbeta  = lambda beta : (lambda _prec,_rec: (1+beta**2)*(_prec*_rec)/(((beta**2)*_prec) + _rec))
freqs = np.sum(emmis,0)/np.sum(emmis)
wtdprec = np.matmul(freqs,prec.T)
wtdrec = np.matmul(freqs,rec.T)
fscores = [
    [0.5,0.5],
    [1,1],
    [2,2]
]
for i in range(len(fscores)):
    score,_ = fscores[i]
    fscores[i][1] = fbeta(score)(wtdprec,wtdrec)
np.savetxt('fscores.csv',np.array(fscores))