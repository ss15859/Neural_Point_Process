import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath('../Simulation'))

from simulation import generate_catalog

import ETAS

################# Simulate Data

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
M_train = np.array(synthetic['magnitude'])


##################### Maximum Likelihood

MLE = ETAS.maxlikelihoodETAS(T_train[:50],M_train[:50],M0=M0)

MLE['beta'] = 1/(M_train-M0).mean()

print(MLE)

T_test = ETAS.datetime64_to_days(test['time'])[:200]
M_test = np.array(test['magnitude'])[:200]

maxtime = T_test.max()

ETASLL = ETAS.marked_likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

ETASLLtemp = ETAS.likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

ETASLLmark = ETAS.mag_likelihood(M_test,params=MLE,Mcut=M0)

print(ETASLLmark+ETASLLtemp)

print(ETASLL)


############ Make residual plots

x, y  = ETAS.generate_residuals(T_test,M_test,params=MLE,Mcut=MLE['M0'])

plt.plot(x,y,color='green',label = 'ETAS')
plt.plot(np.linspace(0,y[-1]),np.linspace(0,y[-1]),color = 'r',label = 'Poisson',linestyle='--')
plt.xlabel('Transformed time')
plt.ylabel('Cumulative number')
plt.legend()
plt.show()



x, y  = ETAS.generate_residuals(T_train,M_train,params=MLE,Mcut=MLE['M0'])

plt.plot(x,y,color='green',label = 'ETAS')
plt.plot(np.linspace(0,y[-1]),np.linspace(0,y[-1]),color = 'r',label = 'Poisson',linestyle='--')
plt.xlabel('Transformed time')
plt.ylabel('Cumulative number')
plt.legend()
plt.show()


############ Daily Forecasting

unit  = 1
forcastETAS = ETAS.daily_forecast(T_test,M_test,ndays=int(np.floor((T_test[-1]-T_test[0])/unit)),repeats=3,params=MLE,hours=unit,Mcut=M0)
aveETAS = np.mean(forcastETAS,axis=1)

true = ETAS.bin_times(T_test,hours=unit)

plt.scatter(range(len(true)),np.log(true))
plt.scatter(range(len(aveETAS)),np.log(aveETAS))
plt.show()