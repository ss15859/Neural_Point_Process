import csv
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('../Simulation'))

from simulation import generate_catalog

import ETAS

from NeuralPP import NPP

#################################################################### Reading Data

Amatrice = pd.read_csv('~/PhD/Amatrice_tests/Amatrice_CAT5.v20210504.csv')

Amatrice['datetime'] = pd.to_datetime(Amatrice[['year', 'month', 'day', 'hour', 'minute','second']])
Amatrice['time'] = (Amatrice['datetime']-Amatrice['datetime'][0])/ pd.to_timedelta(1, unit='H')


Amatrice = Amatrice[['time','mw']]
Amatrice = Amatrice.dropna()

timeupto = int(sys.argv[1])
Mcut = float(sys.argv[2])
M0= 0.3
M0pred = Mcut

sub_catalog = Amatrice[ Amatrice.time<4800]
sub_catalog = Amatrice[Amatrice.mw > Mcut]

times = np.array(sub_catalog['time'])
mags = np.array(sub_catalog['mw'])


T_train = times[times<timeupto]
M_train = mags[times<timeupto]

T_test = times[times>=timeupto]
M_test = mags[times>=timeupto]


###################################################### Maximum Likelihood training

filename = 'data/paramsMcut-'+str(Mcut)+'_timeupto:' + str(timeupto)+ '.csv'

########### ETAS

if not os.path.isfile(filename):

    print('Findling MLE parameters')

    MLE = ETAS.maxlikelihoodETAS(T_train,M_train,M0=M0)

    MLE['beta'] = 1/(M_train-M0).mean()

    with open(filename, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in MLE.items():
           writer.writerow([key, value])


else:
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        MLE = dict(reader)
        MLE = {k:float(v) for k, v in MLE.items()}


print(MLE)        

############# Neural Network

checkpoint = 'checkpoint'+str(Mcut)+str(timeupto)

if not os.path.isfile('./checkpoints/'+ checkpoint+'.index'):
    npp = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=400,batch_size=256).save_weights(checkpoint)

else:
    npp = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).load_weights(checkpoint)

    ######################### Poisson

poissMLE = 1/np.ediff1d(T_train[M_train>=M0pred]).mean()

LLpoiss = (len(T_test[M_test>=M0pred])*np.log(poissMLE) - (T_test[-1]-T_test[0])*poissMLE)/len(T_test[M_test>=M0pred])


############################################################### Evaluating Likelihood


maxtime = T_test.max()

ETASLL = ETAS.marked_likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE)

ETASLLtemp = ETAS.likelihood(T_test,M_test,maxtime=maxtime,Mcut = M0,params = MLE) - LLpoiss

ETASLLmark = ETAS.mag_likelihood(M_test,params=MLE,Mcut=M0)

print('ETAS Mark Likelihood:  '+str(ETASLLmark))
print('ETAS Temporal Likelihood:   ' + str(ETASLLtemp))
print('ETAS Total Likelihood:   ' +str(ETASLL))
# print('Check is the same:   ' + str(ETASLLmark+ETASLLtemp))


npp.set_test_data(T_test,M_test).predict_eval()

NNLL = npp.LL.mean() + npp.LLmag.mean()

NNLLtemp = npp.LL.mean() - LLpoiss

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

# unit  = 24
days = 24
# # days  = int(np.floor((T_test[-1]-T_test[0])/unit))

# forcastETAS = ETAS.daily_forecast(T_test,M_test,ndays=days,repeats=2,params=MLE,hours=unit,Mcut=M0)
# aveETAS = np.mean(forcastETAS,axis=1)
# true = ETAS.bin_times(T_test,hours=unit,ndays=days)


# forcastN = npp.daily_forecast(T_test,M_test,ndays=days,repeats=2,M0=M0,time_step=20,hours=unit)

# ave = np.mean(forcastN,axis=1)
# # ave = forcastN

# plt.scatter(range(len(true[:days])),np.log(true[:days]),label = 'true')
# plt.scatter(range(len(aveETAS)),np.log(aveETAS),label='ETAS')
# plt.scatter(range(len(ave)),np.log(ave),label='NN')
# plt.legend()
# plt.show()

aveETAS = np.zeros(days)
ave= np.zeros(days)
true= np.zeros(days)

####################################################################### Generate output file 

d = {'aveETAS': aveETAS, 'aveNN':ave,'true': true, 'NNLL': np.repeat(NNLL,days),'NNLLtemp':np.repeat(NNLLtemp,days),'NNLLmark':np.repeat(NNLLmark,days), 'ETASLL':np.repeat(ETASLL,days),'ETASLLtemp':np.repeat(ETASLLtemp,days),'ETASLLmark':np.repeat(ETASLLmark,days)}

D = pd.DataFrame(d)

D.to_csv('data/resultsMcut-'+str(Mcut)+'_timeupto:' + str(timeupto)+ '.csv')

