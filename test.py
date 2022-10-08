import torch
import numpy as np
a,b=[[0.2,0.1,0.7],[0.01,0.92,0.07]],[[1,0,0],[0,1,0]]
a = torch.tensor(np.array(a))
b = torch.tensor(np.array(b))


from tagger import Network
m = Network([1,2,3,3],True)
print(m.loss(a,b))