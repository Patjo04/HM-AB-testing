# Written by H&M 2025-01-31
import pandas as pd
import numpy as np
import scipy.stats as st

from scipy.stats import ttest_ind_from_stats
# from AAlib.utils import get_means_vars

from scipy.special import gammaln
from scipy.optimize import root_scalar

from functools import partial

def reqSS_means(mean, var, mde = 0.01, alpha = 0.05, beta = 0.2, direction = "two-sided"):
  if abs(mean)<1e-10:
    # if mean is 0, the effect is compared to 1
    mean = 1
  if direction == "two-sided":
    return int( # Must add ceil for upper approximations.
        2*var*(
        (st.norm.ppf(1.-alpha/2) + st.norm.ppf(1.-beta))/(mean*mde) # coefficient*sigma^2/(my*mde)^2
        )**2)+1 # Can be negative, is an absolute value.
  if direction in ["greater", "less"]:
    return int(
        2*var*(
        (st.norm.ppf(1.-alpha) + st.norm.ppf(1.-beta))/(mean*mde)
        )**2)+1

def ttest(control: pd.DataFrame, treatment: pd.DataFrame, directionality = "two-sided"):
    """
    :control, treatment: dataframes with structure
                         - each row - new experiment (index of the dataframe - id of experiment)
                         - columns - mean_value, var_value, no_sample
    :returns: pvalue for each experiment
    """
    pvals = ttest_ind_from_stats(
        control.mean_value, np.sqrt(control.var_value), control.no_sample,
        treatment.mean_value, np.sqrt(treatment.var_value), treatment.no_sample,
        alternative = directionality
    ).pvalue
    return pvals