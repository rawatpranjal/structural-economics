#!/usr/bin/env python
# coding: utf-8

# # Data and Packages

# In[77]:


# Load packages
import numpy as np
import scipy as sp


# In[78]:


# Data
s = np.array([0.15,0.15,0.3,0.3]) # Market shares, 0.1 for outside good
m = 0.5 # m = (p-c)/p  => c = p(1-m) # price-cost margin for 1st firm (for calibration)
p = np.array([1,1,1,1]) # pre-merger prices
p2f = np.array([1,2,3,4]) # product to firm mapping
P = 1 # market price
Q = 1 # market quantity


# # Functions

# In[79]:


def ownershipMatrix(p2f):
    '''Converts a Jx1 vector mapping product-to-firms into JxJ Ownership matrix Ω
        Ω_{i,j} = 1 if the same firm produces product i and j, else 0. '''
    J = len(p2f) # Number of products
    F = len(np.unique(p2f)) # Number of firms
    Ω = np.zeros((J,J)) # Ownership matrix

    for i in range(J):
        for j in range(J):
            if p2f[i] == p2f[j]: # firm producing product i is same firm that produces product j
                Ω[i,j] = 1

    #print('No. of products:',J)
    #print('No. of firms:',F)
    return  Ω

Ω = ownershipMatrix(p2f)

# In[80]:


def calibrateLogit(m,s,p,p2f):
    '''Input: margin, shares (quantities),prices, product-to-firm mapping
    Output: α: Calibrated price-coeff, ξ: mean non-price values, mc: marginal cost,
    type_j: Nocke Shutz types, type: firm type, dqdp: demand derivatives, div: diversion matrix'''

    Ω = ownershipMatrix(p2f)
    J = len(p2f)
    c1 = p[0]*(1-m) # Cost of 1st firm

    # Generate Cross price derivatives
    temp = -np.outer(s,s) #tcrossprod
    np.fill_diagonal(temp, s*(1-s))

    # Calculate α from the demand
    if 1==len(p2f[p2f==1]):
        α = -1/(1-s[0])/(p[0]-c1)
    else:
        pass

    # Cross price derivatives
    dqdp = temp*α

    # Marginal costs
    c = p + np.dot(np.linalg.inv(Ω*dqdp.T),s)

    # Diversion Matrix D[k,j] = s_k / (1-sj) and -1 on diagonal
    div = np.multiply(s,1/(1-s).reshape(-1,1))
    np.fill_diagonal(div, -1)

    # Mean Valuations
    ξ = np.log(s/(1-np.sum(s)))-α*p

    # Type
    type_j = np.exp(ξ-α*c)
    Type = np.bincount(p2f-1, weights=type_j)

    return α, ξ, c, type_j, Type, dqdp, div

α, ξ, c, type_j, Type, dqdp, div = calibrateLogit(m,s,p,p2f)

print(α, ξ)
print(dqdp)
print(div)


# In[81]:


def foc(p,c,s,dqdp,Ω):
    '''Checks the FOC conditions in the Bertrand-Nash price competition, for any generic demand derivatives.
    Inputs: Vectors of prices, marginal costs, quantities (shares), demand derivative matrix, ownership (structure) matrix.
    Output: Upward Pricing Pressure, or the value of FOC for each product
    '''
    FOC = -p + c - np.dot(np.linalg.inv(Ω*dqdp.T),s)
    return np.round(FOC,4)

foc(p,c,s,dqdp,Ω)


# In[82]:


def quant_logit(p, α, ξ):
    '''Input: Vector of Prices p, price elasticity α and vector of unobserved quality ξ
    Output: Vector of shares (quantities) '''
    s = np.exp(p*α+ξ) * (1/(1+np.sum(np.exp(p*α+ξ))))
    return s

quant_logit(p, α, ξ)


# In[83]:


def dqdp_logit(p,α,ξ):
    '''Demand derivatives for Logit Demand
    Input: prices, price elasticity, unobserved quality
    Output: JxJ demand derivative matrix'''
    s = quant_logit(p, α, ξ)
    temp = -np.outer(s,s) #tcrossprod
    np.fill_diagonal(temp, s*(1-s))
    dqdp = α * temp
    return dqdp

dqdp_logit(p,α,ξ)


# In[84]:


foc(p,c,quant_logit(p, α, ξ),dqdp,Ω)


# In[85]:


def foc_logit(p,c,α,ξ,p2f):
    '''Check FOC for Logit'''
    Ω = ownershipMatrix(p2f)
    dqdp = dqdp_logit(p,α,ξ)
    s = quant_logit(p,α,ξ)
    FOC = foc(p,c,s,dqdp,Ω)
    return FOC

foc_logit(p,c,α,ξ,p2f)


# # Merger Simulation 1: Firm1 with Firm 2

# In[86]:


# Step 1: Calibrate to obtain Structural Parameters
α, ξ, c, type_j, Type, dqdp, div = calibrateLogit(m,s,p,p2f)


# In[87]:


# Step 2: Check the FOC conditions
foc_logit(p,c,α,ξ,p2f)


# In[88]:


# Step 3: Create new ownership matrix relating to post-merger situation
p2f_new = np.array([1,1,3,4])


# In[89]:


# Step 4: Calculate value of FOC with post-merger structure with pre-merger prices
# This is upward pricing pressure
foc_logit(p,c,α,ξ,p2f_new)


# In[90]:


# Step 5: Calculate the Post Merger Prices
p_new = sp.optimize.fsolve(foc_logit, x0=np.array([0,0,0,0]), args=(c,α,ξ,p2f_new))
print(p_new)


# In[91]:


# Step6: Calculate new shares/quantities
q_new = quant_logit(p_new, α, ξ)
print(q_new)


# # Merger Simulation 2: Reduced Costs

# In[92]:

p_new = sp.optimize.fsolve(foc_logit, x0=np.array([0,0,0,0]), args=(c*np.array([0.9,0.9,1,1]),α,ξ,p2f_new))
print(p_new)

q_new = quant_logit(p_new, α, ξ)
print(q_new)


# # Merger Simulation 3: Collusion

# In[93]:


p_new = sp.optimize.fsolve(foc_logit, x0=np.array([1,2,1,1]), args=(c,α,ξ,np.array([1,1,1,1])))
print(p_new)

q_new = quant_logit(p_new, α, ξ)
print(q_new)

