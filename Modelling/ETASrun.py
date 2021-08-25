import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import scipy.integrate as integrate

sys.path.append(os.path.abspath('../Simulation'))

from simulation import generate_catalog

import ETAS

from NeuralPP import NPP

################## Check whether to do new run

if len(sys.argv)>1:
    firstarg=sys.argv[1]
else:
    firstarg = 'none'

#################################################################### Simulating Data

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

if not os.path.isfile('./data/synthetic_catalog.csv') or firstarg == 'new_run':
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

    synthetic[["time", "magnitude"]].to_csv('data/synthetic_catalog.csv')
    test[["time", "magnitude"]].to_csv('data/synthetic_test_catalog.csv')

else:
    synthetic = pd.read_csv('data/synthetic_catalog.csv')
    synthetic['time'] = pd.to_datetime(synthetic['time'])

    test = pd.read_csv('data/synthetic_test_catalog.csv')
    test['time'] = pd.to_datetime(test['time'])

T_train = ETAS.datetime64_to_days(synthetic['time'])
M_train = np.array(synthetic['magnitude'])

T_test = ETAS.datetime64_to_days(test['time'])
M_test = np.array(test['magnitude'])


###################################################### Maximum Likelihood training

########### ETAS

if not os.path.isfile('./data/params.csv') or firstarg == 'new_run':

    print('Findling MLE parameters')

    MLE = ETAS.maxlikelihoodETAS(T_train,M_train,M0=M0)

    MLE['beta'] = 1/(M_train-M0).mean()

    with open('data/params.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in MLE.items():
           writer.writerow([key, value])


else:
    with open('data/params.csv') as csv_file:
        reader = csv.reader(csv_file)
        MLE = dict(reader)
        MLE = {k:float(v) for k, v in MLE.items()}


print(MLE)        

############# Neural Network

if not os.path.isfile('./checkpoints/my_checkpoint.index'):
    npp = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=400,batch_size=256).save_weights('./my_checkpoint')

else:
    npp = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).load_weights('./my_checkpoint')

############################################################### Evaluating Likelihood


maxtime = T_test.max()

ETASLL = ETAS.marked_likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

ETASLLtemp = ETAS.likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

ETASLLmark = ETAS.mag_likelihood(M_test,params=MLE,Mcut=M0)

print('ETAS Mark Likelihood:  '+str(ETASLLmark))
print('ETAS Temporal Likelihood:   ' + str(ETASLLtemp))
print('ETAS Total Likelihood:   ' +str(ETASLL))
# print('Check is the same:   ' + str(ETASLLmark+ETASLLtemp))


npp.set_test_data(T_test,M_test).predict_eval()

NNLL = npp.LL.mean() + npp.LLmag.mean()

NNLLtemp = npp.LL.mean()

NNLLmark = npp.LLmag.mean()

print('NN Mark Likelihood:  '+str(NNLLmark))
print('NN Temporal Likelihood:   ' + str(NNLLtemp))
print('NN Total Likelihood:   ' +str(NNLL))

################################################################## Make residual plots

# y = range(1,len(npp.Int_lam)+1)
# t = np.cumsum(npp.Int_lam)

# lam_int, yETAS  = ETAS.generate_residuals(T_test,M_test,params=MLE,Mcut=MLE['M0'])

# plt.plot(t,y,color='black',label = 'NN')
# plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
# plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'Poisson',linestyle='--')
# plt.xlabel('Transformed time')
# plt.ylabel('Cumulative number')
# plt.title('Residual plot on Test Data')
# plt.legend()
# plt.show()


# npp.eval_train_data()

# y = range(1,len(npp.Int_lam_train)+1)
# t = np.cumsum(npp.Int_lam_train)

# lam_int, yETAS  = ETAS.generate_residuals(T_train,M_train,params=MLE,Mcut=MLE['M0'])    

# plt.plot(t,y,color='black',label = 'NN')
# plt.plot(lam_int,yETAS,color='green',label = 'ETAS')
# plt.plot(np.linspace(0,t[-1]),np.linspace(0,t[-1]),color = 'r',label = 'Poisson',linestyle='--')
# plt.xlabel('Transformed time')
# plt.ylabel('Cumulative number')
# plt.title('Residual plot on Training Data')
# plt.legend()
# plt.show()


###################################################################### Daily Forecasting

# unit  = 1
# days = 30
# # days  = int(np.floor((T_test[-1]-T_test[0])/unit))

# forcastETAS = ETAS.daily_forecast(T_test,M_test,ndays=days,repeats=2,params=MLE,hours=unit,Mcut=M0)
# aveETAS = np.mean(forcastETAS,axis=1)
# true = ETAS.bin_times(T_test[T_test<=days],hours=unit)


# forcastN = npp.daily_forecast(T_test,M_test,ndays=days,repeats=2,M0=M0,time_step=20,hours=unit)

# ave = np.mean(forcastN,axis=1)
# # ave = forcastN

# plt.scatter(range(len(true[:days])),np.log(true[:days]),label = 'true')
# plt.scatter(range(len(aveETAS)),np.log(aveETAS),label='ETAS')
# plt.scatter(range(len(ave)),np.log(ave),label='NN')
# plt.legend()
# plt.show()



################################################################# mean of Mag Distribution


# x = np.linspace(0,10,100)
# index_len = len(T_test)
# Mcut=M0
# mean = np.zeros(index_len)


# for index in range(21,26):
#     index = int(index)
#     T = T_test[:index]
#     M = M_test[:index]
#     new_time = T_test[index]

#     yNN = np.array([npp.magdensfunc(x=i,hist=[T,M],new_time=new_time) for i in x])

#     # print(np.dot(x,yNN))
#     mean[index] = integrate.quad(lambda i: npp.magdensfunc(x=i,hist=[T,M],new_time=new_time), 0, 10)[0]

# # y = ETAS.sGR(x,true_parameters,Mcut)*np.exp((Mcut-true_parameters['M0'])*true_parameters['beta'])
# plt.plot(range(len(mean)),mean,label = 'NN Magnitude Distribution')
# plt.plot(range(len(mean)),np.repeat(1/true_parameters['beta'],len(mean)),label = 'Gutenberg-Richter')
# # plt.hist(M,density=True,bins=20,alpha=0.5)
# plt.legend()
# plt.show()

