import numpy as np
a = []
for i in range(3):
    a.append(np.random.random((4,10)))
print((np.mean(a,0)).shape)