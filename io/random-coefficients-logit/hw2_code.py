import pandas as pd
import numpy as np
from pandas.testing import assert_index_equal


def sigma_mc(delta_est:pd.Series):
    """
    Implements a Monte-Carlo estimator of sigma_{jt}
    delta_est must be a Series indexed by the same MultiIndex((cdid,prodid)) as df
    """
    assert_index_equal(delta_est.index, cust_idutils.index)

    normlzr = 1 + np.exp(cust_idutils.add(delta_est, axis='index')).groupby(by='cdid').sum()
    return np.exp(cust_idutils.add(delta_est, axis='index')).divide(
        normlzr,
        axis='index'
    ).mean(axis=1)

T = lambda expdel: expdel*shares/sigma_mc(np.log(expdel))

def est_delta(delta0:pd.Series, tol:float=1e-6):
    """
    Iterates the contraction mapping until the error between shares and share estimates
    is smaller than tol, for the standard euclidean distance, with the sampling measure.
    """
    assert tol>0
    expdel = np.exp(delta0)
    while True:
        expdel_next = T(expdel)
        err = np.linalg.norm(expdel_next - expdel, np.Inf)
        print(expdel)
        if err < tol:
            break
        expdel = expdel_next

    return np.log(expdel)


rawdata = pd.read_csv("mumat.csv")
rawdata.set_index(keys=['cdid','prodid'], inplace=True)
#print(rawdata)
shares = rawdata['s_jt']
cust_idutils = rawdata[rawdata.columns[1:]]
print(cust_idutils)

delta0 = shares
delta_est = est_delta(delta0, tol=1e-12)
delta_est.name = "Estimated mean valuations"

print(delta0)
print(delta_est)
