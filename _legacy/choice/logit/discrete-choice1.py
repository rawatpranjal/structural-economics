import numpy as np
X = np.array(range(100))
D = np.array(range(100))

def u(x,d):
    return (x+d)-0.01*(x+d)**2

def e(d):
    mean = d-0.01*d**2
    return np.random.gumbel(loc=mean, scale=1.0, size=1)

def d(x,e):
    max = -500000
    for d in D:
        utility = u(x,d)+e(d)
        if utility>=np.array(max):
            d_star=d
            max=utility
    return d_star, max


N = 10000
sample = [d(0,e) for i in range(N)]

import seaborn as sns
sns.histplot(sample)