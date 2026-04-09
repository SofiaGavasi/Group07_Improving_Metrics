from __future__ import annotations

from typing import Any, Dict
import numpy as np



def polynomial_kernel(x: np.ndarray,y: np.ndarray):
    d = x.shape[1] # kernel on feature dimension
    return ((1/d)*(x @ y.T)+ 1 )**3

def compute_kid(real_features: np.ndarray, fake_features: np.ndarray):
    m= real_features.shape[0] # comparing based on samples
    n= fake_features.shape[0]

    Kxx = polynomial_kernel(real_features,real_features)
    sum_Kxx=  np.sum(Kxx) - np.trace(Kxx) # we need this so it is not biased
    Kyy = polynomial_kernel(fake_features,fake_features)
    sum_Kyy=  np.sum(Kyy) - np.trace(Kyy)

    sum_Kxy = np.sum(polynomial_kernel(real_features,fake_features))

    kid = (1/(m*(m-1)))*sum_Kxx +(1/(n*(n-1)))*sum_Kyy  - 2*(1/(n*m))*sum_Kxy

    return float(kid)

    ## TO DO: 
    #   - add multible runs, so it calculates for many subsets
    #   - add std 
    #   - compare results with library 



    #  return keys: {"mean": float, "std": float}
    #raise NotImplementedError("TODO: implement compute_kid")
