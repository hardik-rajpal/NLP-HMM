import numpy as np
import matplotlib.pyplot as plt
# a,b=[[0.2,0.1,0.7],[0.01,0.92,0.07]],[[1,0,0],[0,1,0]]
# a = torch.tensor(np.array(a))
# b = torch.tensor(np.array(b))


# from tagger import Network
# m = Network([1,2,3,3],True)
# print(m.loss(a,b))
arr = np.random.random((12,12))
plt.imshow(arr,cmap='hot',interpolation='nearest')
plt.text(4,-1,f'Range:({np.round(np.min(arr),2)},{np.round(np.max(arr),2)})')
plt.show()