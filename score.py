import numpy as np
scores = np.loadtxt('results/perposmetrics.csv')
freqs = np.loadtxt('trained/tagfreqs.csv')
freqs = freqs.tolist()
freqs.append(0)
freqs = np.array(freqs)
freqs/=np.sum(freqs)
prec = scores[0,:]/100
rec = scores[1,:]/100
print(rec,prec)
beta = 0.5
fbeta  = lambda beta : (lambda _prec,_rec: (1+beta**2)*(_prec*_rec)/(((beta**2)*_prec) + _rec))
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
np.savetxt('results/fscores.csv',100*np.array(fscores),fmt='%.2f')
print('helooo')