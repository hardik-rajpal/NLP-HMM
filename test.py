import torch
import numpy as np
a = np.array([[0.1,0.8,0.1,0],[0.7,0.2,0.1,0]])
b = np.array([0,1])

from tagger import Network
m = Network([1,2,3])
print(m.loss(torch.tensor(a),torch.tensor(b)))