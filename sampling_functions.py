from scipy.stats import norm, poisson, skellam
import numpy as np

def norm_sample(loc, scale, min, max=[]):
    dist = norm(loc=loc, scale=scale)
    sample = dist.rvs(1)[0] 
    
    if not max:
        sample = np.max([sample,min])  
    else:
        sample = np.max([np.min([sample,max]),min])
        
    return sample

def truncated_poisson(loc, mu, min):
    #dist = poisson(mu=mu)
    sample = poisson.rvs(mu=mu, loc=loc, size=1)[0] 
    return sample if sample > min else min 

def truncated_skellam(loc, mu1, mu2, min):
    sample = skellam.rvs(mu1=mu1, mu2=mu2, loc=loc, size=1)[0] 
    return sample if sample > min else min 
