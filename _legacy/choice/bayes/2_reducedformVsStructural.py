# %% [markdown]
# ### <a id='toc1_1_1_'></a>[Learning how to learn](#toc0_)

# %% [markdown]
# **Table of contents**<a id='toc0_'></a>    
# - [Learning how to learn](#toc1_1_1_)    
#     - [Part 1: Reduced Form Approach](#toc1_1_1_1_)    
#     - [Structural Logit Model](#toc1_1_2_)    
# 
# <!-- vscode-jupyter-toc-config
# 	numbering=false
# 	anchor=true
# 	flat=false
# 	minLevel=1
# 	maxLevel=6
# 	/vscode-jupyter-toc-config -->
# <!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# %% [markdown]
# #### <a id='toc1_1_1_1_'></a>[Part 1: Reduced Form Approach](#toc0_)
# 
# There are two cages - A and B. The probabilty of drawing ball N from A is 2/3 and from B is 1/3. First, with prior p we choose a cage. Then we draw with replacement 6 balls from it and record the fraction that are N. We repeat these trials many times. The (prior, n_draws, sample_size) constitute the environment and (cages, draws) constitute the training data. 
# 
# Problem 1: Build a machine learning algoithm than predicts cages from draws. Observe the test accuracy - i.e the P(predict=A|cage=A). And compare this with Bayes rule. Bayes rule when $P(A)$ is prior and $N$ is number of balls observed distributed by $N|A \sim BIN(6, 2/3)$ and $N|B \sim BIN(6, 1/3)$:
# 
#  $P(A|N=n) = \frac{P(N=n|A)P(A)}{P(N=n|A)P(A)+P(N=n|B)P(B)}$
# 
# Problem 2: Build a model of subjective beliefs and 

# %%
import numpy as np
import pandas as pd

# %%
def training_data_generator(n_draws, prior, sample_size):
    import numpy as np
    cages = np.random.binomial(1, prior, size=sample_size)
    drawA = np.random.binomial(n_draws, 2/3, size=sample_size)
    drawB = np.random.binomial(n_draws, 1/2, size=sample_size)
    draws = np.where(cages==1, drawA, drawB)
    return cages, draws 

cages, draws  = training_data_generator(6, 1/2, 10000)

# %%
def preprocessing(cages, draws):
    import pandas as pd
    df = pd.DataFrame(cages, columns=['cage'])
    df['draws'] = draws
    df.head()
    x = df.drop('cage', axis = 1)
    y = df['cage'].astype('int')
    return x, y

x, y = preprocessing(cages, draws)

# %%
def machineLearning(n_draws, prior, sample_size):
    """Input: Priors, Draws, Sample Size
    Output: Table of Cross-validated Test-Accuracy of various machine learning classifiers"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from prettytable import PrettyTable
    from PIL import Image, ImageDraw, ImageFont
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.neural_network import MLPClassifier
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')
    np.random.seed(3293423)

    cages, draws  = training_data_generator(n_draws, prior, sample_size)
    x, y = preprocessing(cages, draws)
    
    model1=LogisticRegressionCV()
    model2=RandomForestClassifier(n_estimators=100)
    model3=GradientBoostingClassifier(n_estimators=100)
    model4=MLPClassifier(max_iter=5000,activation='tanh')
    model5=XGBClassifier(max_depth=2, n_estimators=100)
    model6=CatBoostClassifier(max_depth=2, n_estimators=100, verbose=0)

    table = PrettyTable()
    table.field_names = ['Estimator', 'Avg Cross-Validated Test-Accuracy']
    table.add_row(['Logistic Regression',cross_val_score(model1, x, y, cv=10).mean()])
    table.add_row(['Random Forest Classifier',cross_val_score(model2, x, y, cv=10).mean()])
    table.add_row(['Gradient Boosting Classifier',cross_val_score(model3, x, y, cv=10).mean()])
    table.add_row(['Neural Network Classifier',cross_val_score(model4, x, y, cv=10).mean()])
    table.add_row(['XGBoost Classifier',cross_val_score(model5, x, y, cv=10).mean()])
    table.add_row(['CatBoost Classifer',cross_val_score(model6, x, y, cv=10).mean()])
    table.float_format = '0.3'
    print(table)

machineLearning(6, 2/3, 1000)


# %%
def bayesRule(n, p, N):
    """Input: Number of balls drawn in current trial, prior prob of cage A, total number of draws in every trial"""
    """Output: Prob(A|N=n) or Posterior Prob of A given that n draws were seen at latest trial"""
    from scipy.stats import binom
    pA = p
    pB = 1-p
    pnA = binom.pmf(n, N, 2/3)
    pnB = binom.pmf(n, N, 1/2)
    pn = pnA*pA+pnB*pB
    pAn = pnA*pA/pn
    return pAn

p1 = bayesRule(5, 1/3, 6)
p2 = bayesRule(4, 1/2, 6)
p3 = bayesRule(3, 2/3, 6)
print(p1,p2,p3)

# %%
def posteriorOdds(draws, priors):
    rows = []
    for prior in priors: 
        for n in range(draws):
            pAn = bayesRule(n, prior, draws)
            rows.append([pAn, prior, n, draws])
    df=pd.DataFrame(rows, columns=['posterior', 'prior', 'draws of N', 'total draws'])
    df.prior = df.prior.astype(str)
    return df

def plotOdds(df):
    import plotly.express as px
    px.line(df, x="draws of N", y="posterior", color = "prior", markers = True).show() 


# Show plot 
df = posteriorOdds(6, [1/3, 1/2, 2/3])
plotOdds(df)


# %%
def compareMLwithBayes(n_draws, prior, sample_size):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from prettytable import PrettyTable

    model1=LogisticRegressionCV()
    model2=RandomForestClassifier(max_depth=2)
    model3=CatBoostClassifier(max_depth=2, verbose=0)
    cages, draws  = training_data_generator(n_draws, prior, sample_size)
    x, y = preprocessing(cages, draws)
    table = PrettyTable()
    table.field_names = ['#of N', 'Bayes Rule: P(A|n)', 'Logistic Regression: P(A|n)', 'Random Forest: P(A|n)', 'CatBoost Classifier: P(A|n)']
    for n in range(n_draws):
        MLprob1 = model1.fit(x,y).predict_proba(np.array([n]).reshape(1, -1))[0][1]
        MLprob2 = model2.fit(x,y).predict_proba(np.array([n]).reshape(1, -1))[0][1]
        MLprob3 = model3.fit(x,y).predict_proba(np.array([n]).reshape(1, -1))[0][1]
        Bayesprob = bayesRule(n, prior, 6)
        table.add_row([n, Bayesprob, MLprob1, MLprob2, MLprob3])
    table.float_format = '0.3'
    print(table)

compareMLwithBayes(6, 1/2, 1000)


# %% [markdown]
# #### <a id='toc1_1_2_'></a>[Part 2: Structural Logit Model](#toc0_)
# 
# $\pi_A$ is prior probability of choosing cage A, $A$ is event that cage chosen is A, $n$ is number of draws of ball N, $p_A$ is probability of drawing ball N from cage A, $p_B$ is prob of drawing ball N from cage B. 
# 
# $P(A|n,\pi_A) = \frac{1}{1+exp(a+bn+c(log(1-\pi_A)-log(\pi_A))}$
# 
# Bayes Rule: 
# - $a = 6(\log(1-p_B)-\log(1-p_A))$
# - $b = log(p_B/(1-p_B))-log(p_A/(1-p_A))$
# - $c = 1$
# 

# %%
def trueParam():
    p_A = 2/3
    p_B = 1/2
    a = 6 * (np.log((1-p_B)/(1-p_A)))
    b = np.log(p_B*(1-p_A)/p_A/(1-p_B))
    c = 1
    return p_A, p_B, a, b, c

p_A, p_B, a, b, c = trueParam()
print('True params:', a,b,c)
    
def logit(n, prior, a, b, c):
    denominator = 1 + np.exp(a+b*n+c*(np.log((1-prior)/prior)))
    return np.power(denominator,-1)

def logitBayes(n, prior):
    p_A, p_B, a, b, c = trueParam()
    return logit(n, prior, a, b, c)

import plotly.express as px
px.line(np.array([[logitBayes(n, prior) for n in range(6)] for prior in [2/3,1/2,1/3]]).T).show()

# %%
def trainStructuralLogit(N, priors, n_draws):
    p1, p2, p3 = priors
    cages, draws  = training_data_generator(n_draws, p1, N)
    x1, y1 = preprocessing(cages, draws)
    cages, draws  = training_data_generator(n_draws, p2, N)
    x2, y2 = preprocessing(cages, draws)
    cages, draws  = training_data_generator(n_draws, p3, N)
    x3, y3 = preprocessing(cages, draws)
    x1 = np.r_[x1,x2,x3].reshape(-1)
    x2 = np.r_[1/3*np.ones(N), 1/2*np.ones(N), 2/3*np.ones(N)]
    y = np.r_[y1,y2,y3]

    import torch 
    torch.manual_seed(3)
    y = torch.tensor(y, dtype=torch.float)
    x1 = torch.tensor(x1, dtype=torch.float)
    x2 = torch.tensor(x2, dtype=torch.float)

    class logit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(()))
            self.b = torch.nn.Parameter(torch.randn(()))
            self.c = torch.nn.Parameter(torch.randn(()))

        def forward(self, n, pi):
            return torch.pow(1+torch.exp(self.a + self.b * n + self.c * torch.log((1-pi)/pi)),-1)

    model = logit()
    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(50000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x1,x2)

        # Compute and print loss
        loss = criterion(y_pred, y)
        #if t % 100 == 99:
            #print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return x1, x2, y, model
    
x1, x2, y, model = trainStructuralLogit(100, [1/3,1/2,2/3], 6)

# %%
# Estimated Parameters
print(model.a.item(), model.b.item(), model.c.item())

# %% [markdown]
# N = 100

# %%
def compareStructuralLogitwithBayes(N, priors, n_draws):
    import torch
    x1, x2, y, model = trainStructuralLogit(N, priors, n_draws)
    from prettytable import PrettyTable
    for pi in priors:
        table = PrettyTable()
        table.field_names = ['#of N', 'Prior', 'Bayes Rule: P(A|n)', 'Structural Logit: P(A|n)']
        for n in range(n_draws):
            SLprob = model(torch.tensor(n),torch.tensor(pi)).item()
            Bayesprob = bayesRule(n, pi, 6)
            table.add_row([n, pi, Bayesprob, SLprob])
        table.float_format = '0.3'
        print(table)
    return model.a.item(), model.b.item(), model.c.item()

params = []
a, b, c = compareStructuralLogitwithBayes(100, [1/3,1/2,2/3], 6)
params.append([100, a, b, c])

# %% [markdown]
# N=1000

# %%
a, b, c = compareStructuralLogitwithBayes(1000, [1/3,1/2,2/3], 6)
params.append([1000, a, b, c])

# %% [markdown]
# N=10000

# %%
a, b, c = compareStructuralLogitwithBayes(10000, [1/3,1/2,2/3], 6)
params.append([10000, a, b, c])

# %%
print("Estimated vs True Params: as Training Size increases")
print(np.array(params))
print(a,b,c)


# %%
def posteriorOddsSL(n_draws, priors):
    import torch
    x1, x2, y, model = trainStructuralLogit(10000, priors, n_draws)
    rows = []
    for pi in priors:
        for n in range(n_draws):
            SLprob = model(torch.tensor(n),torch.tensor(pi)).item()
            Bayesprob = bayesRule(n, pi, 6)
            rows.append([n, pi, SLprob, Bayesprob])
    df=pd.DataFrame(np.array(rows), columns=['draws of N', 'prior', 'Structural Logit', 'Bayesprob'])
    df.prior = df.prior.astype(str)
    return df

def plotOddsSL(df):
    import plotly.express as px
    px.line(df, x="draws of N", y="Structural Logit", color = "prior", markers = True).show() 
    px.line(df, x="draws of N", y="Bayesprob", color = "prior", markers = True).show() 


# Show plot 
df = posteriorOddsSL(6, [1/3, 1/2, 2/3])
plotOddsSL(df)


