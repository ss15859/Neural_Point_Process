import csv
import numpy as np
import datetime as dt
import sys
import os
sys.path.append(os.path.abspath('../Simulation'))
from simulation import generate_catalog

import ETAS



start = dt.datetime(1992, 1, 1)
end = dt.datetime(2000, 1, 1)
test_end=dt.datetime(1994,1,1)

M0 = 2
beta = 2.4

true_parameters = {
'mu': 1.2,
'k0': 0.2,
'a': 1.5,
'c': 0.5,
'omega': 1.5,
'tau': np.power(10,3.99),
'M0':M0,
'beta':beta,
}


synthetic = generate_catalog(
    timewindow_start=start,
    timewindow_end=end,
    parameters=true_parameters,
    mc=M0,
    beta_main=beta
)

synthetic = synthetic.sort_values('time')


test = generate_catalog(
        timewindow_start=start,
        timewindow_end=test_end,
        parameters=true_parameters,
        mc=M0,
        beta_main=beta
    )
    
test = test.sort_values('time')


T_train = ETAS.datetime64_to_days(synthetic['time'])
M_train = synthetic['magnitude']


MLE = ETAS.maxlikelihoodETAS(T_train[:50],M_train[:50],M0=M0)

MLE['beta'] = 1/(M_train-M0).mean()

print(MLE)

T_test = ETAS.datetime64_to_days(test['time'])
M_test = test['magnitude']

maxtime = T_test.max()

ETASLL = ETAS.marked_likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

print(ETASLL)
