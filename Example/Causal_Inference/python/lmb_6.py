import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from itertools import combinations 
import plotnine as p

# read data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def read_data(file): 
	return pd.read_stata("https://raw.github.com/scunning1975/mixtape/master/" + file)

def lm_robust(formula, data):
    regression = sm.OLS.from_formula(formula, data = data)
    regression = regression.fit(cov_type="cluster",cov_kwds={"groups":data['id']})
    return regression

lmb_data = read_data("lmb-data.dta")

lmb_data['demvoteshare_c'] = lmb_data['demvoteshare'] - 0.5
# drop missing values
lmb_data = lmb_data[~pd.isnull(lmb_data.demvoteshare_c)]
lmb_data['demvoteshare_sq'] = lmb_data['demvoteshare_c']**2
lmb_subset = lmb_data[lmb_data.demvoteshare.between(.45, .55)]


lm_1 = lm_robust('score ~ lagdemocrat*demvoteshare_c + lagdemocrat*demvoteshare_sq', 
                 data = lmb_subset)
lm_2 = lm_robust('score ~ democrat*demvoteshare_c + democrat*demvoteshare_sq', 
                 data = lmb_subset)
lm_3 = lm_robust('democrat ~ lagdemocrat*demvoteshare_c + lagdemocrat*demvoteshare_sq', 
                 data = lmb_subset)
print("Results based on ADA Scores -- Close Sample with linear and quadratic interactions")
Stargazer([lm_1, lm_2, lm_3])
