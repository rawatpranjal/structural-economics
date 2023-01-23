#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)
df = pd.read_csv('/Users/pranjal/Desktop/Structural-Economics/io/random-coefficients-logit/mumat.csv')

# index of records
ID = np.array(df.index)
ID_idx = ID.shape[0]

# cdid: market id (total 2)
cdid = np.array(df['cdid'])

# prodid: product id (total 9)
prodid = np.array(df['prodid'])

# cdindex: index of last element in market
cdindex = np.searchsorted(cdid, np.unique(cdid))

# market shares for each product j and market t
s = np.array(df['s_jt'])

# Mean-Deviations: μ_ijt for ith customer, for JxT product/markets.
μ = np.array(df.drop(['cdid', 'prodid', 's_jt'], axis = 1))

# Share of the outside good in each market t
s_sum = np.array(df[['s_jt','cdid']].groupby('cdid').sum())
s0 = 1 - s_sum
s0 = np.where(cdid==1, s0[0], s0[1])

# Initial guess for Mean-utilities
δ = np.log(s) - np.log(s0)

# Number of customers, products and markets
N = μ.shape[1]
J = np.unique(prodid).shape[0]
T = np.unique(cdid).shape[0]

def CCP(μ_ijt, μ_irt, δ_jt, δ_rt):
    '''For a market t, given mean valuations and mean deviations of products '''
    try: 
        return (np.exp(δ_jt + μ_ijt)/(1 + np.sum(np.exp(δ_rt + μ_irt))))[0]
    except: 
        return 0

def CCPMatrix(μ, δ):
    '''Return Consumer Choice Probability for each i and product/market'''
    P = np.zeros((ID_idx, N))
    for t in range(1,T+1):
        for j in range(1,J+1):
            idx = np.multiply(cdid==t, prodid==j)
            δ_jt = δ[idx]
            δ_rt = δ[cdid==t]
            for i in range(1,N+1):
                μ_ijt = μ[idx, i-1]
                μ_irt = μ[cdid==t, i-1]
                P[idx, i-1] = CCP(μ_ijt, μ_irt, δ_jt, δ_rt) 
    return P

def σ_jt(P, j, t):
    '''Using CCP return Market share for product j and t'''
    idx = np.multiply(cdid==t, prodid==j)
    if np.mean(P[idx, :])>0:
        return np.mean(P[idx, :])
    else: 
        return 0

def contractionMap(δ, μ, tol=0.000001):
    '''Input: Guess for mean-valuations and mean-deviations for all products and all markets
    Output: Optimal mean-valuations
    '''
    expδ = np.exp(δ)
    error = 1
    cnt = 1
    while error > tol:
        P = CCPMatrix(μ, np.log(expδ))
        for t in range(1,T+1):
            for j in range(1,J+1):
                idx = np.multiply(cdid==t, prodid==j)
                expδ[idx] = expδ[idx]*s[idx]/σ_jt(P, j, t)
                error = np.linalg.norm(expδ[idx]*s[idx]/σ_jt(P, j, t) - expδ[idx])
        cnt = cnt + 1
    return np.log(expδ) # return δ

δ_0 = np.log(s) - np.log(s0) # initial guess
δ = contractionMap(δ_0, μ)
print(δ.reshape(-1,1))

