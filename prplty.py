import numpy as np
p = 0.3
trials = 10000
counts = np.random.binomial(n=10, p=p, size=trials)
print('empirical mean', counts.mean())
print('empirical var', counts.var())