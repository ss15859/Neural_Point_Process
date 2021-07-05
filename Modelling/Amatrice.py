import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
import seaborn as sns

from NeuralPP import NPP

from ETAS import marked_ETAS_intensity, marked_likelihood, movingaverage

import ETAS



Amatrice = pd.read_csv('~/PhD/Amatrice_tests/Amatrice_CAT5.v20210504.csv')

Amatrice['datetime'] = pd.to_datetime(Amatrice[['year', 'month', 'day', 'hour', 'minute','second']])
Amatrice['time'] = (Amatrice['datetime']-Amatrice['datetime'][0])/ pd.to_timedelta(1, unit='H')


Amatrice = Amatrice[['time','mw']]
Amatrice = Amatrice.dropna()

timeupto = 6000
timefrom = 0
M0= 2.5
M0pred = 3.0

time_step=20

# sub_catalog = Amatrice[ Amatrice.time<timeupto]
sub_catalog = Amatrice[Amatrice.mw > M0]



times = np.array(sub_catalog['time'])
mags = np.array(sub_catalog['mw'])

limit = np.linspace(1000,6000,6)

MLE = []
NN = []
poiss=[]

for timeupto in limit:
    
    timeupto = int(timeupto)
    T_train = times[times<timeupto]
    M_train = mags[times<timeupto]

    T_test = times[times>=timeupto]
    M_test = mags[times>=timeupto]
#     T_test = T_test[M_test>=M0pred]
#     M_test = M_test[M_test>=M0pred]

    
    npp1 = NPP(time_step=20,size_rnn=64,size_nn=64,size_layer_chfn=3,size_layer_cmfn=2).set_train_data(T_train,M_train).set_model(0).compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()

    LLNN = npp1.LL[M_test[time_step+1:]>=M0pred].mean()

    filename = '~/PhD/Amatrice_tests/paramsM0:2.500000_timefrom:0_timeupto:' + str(timeupto)+ '.csv'

    params = pd.read_csv(filename)


    groundMLE = params.x[0]
    k0MLE= params.x[1]
    alphaMLE = params.x[2]
    cMLE = params.x[3]
    wMLE = params.x[4]
    betaMLE = params.x[5]
#     M0 = mags.min()
    tau =0


    poissMLE = 1/np.ediff1d(T_train[M_train>=M0pred]).mean()
    
    LLpoiss = (len(T_test[M_test>=M0pred])*np.log(poissMLE) + sum(np.log(ETAS.sGR(M_test,betaMLE,M0pred)))- (T_test[-1]-T_test[0])*poissMLE)/len(T_test[M_test>=M0pred])

    LLMLE = marked_likelihood(T_test,M_test,T_test[-1],groundMLE,k0MLE,alphaMLE,M0pred,cMLE,tau,wMLE,betaMLE)/len(T_test[M_test>=M0pred])

    result1 = pd.DataFrame([LLNN-LLpoiss,LLMLE-LLpoiss,LLpoiss-LLpoiss],columns=['log-likelihood gain'],index=['NN','ETAS','Poison'])
    
    result1

    NNgain = LLNN-LLpoiss

    MLEgain = LLMLE - LLpoiss
    
    poiss = np.append(poiss,LLpoiss)

    NN = np.append(NN,NNgain)

    MLE = np.append(MLE,MLEgain)
    
plt.plot(limit,NN,label='NN')
plt.plot(limit,MLE,label='ETAS')
plt.legend()
plt.xlabel('time from start / hours')
plt.ylabel('log-likelihood gain')
plt.title('log-likelihood gain with increasing training size for M0_train = 2.5, M0_test = 3.0')
# plt.savefig('loglik_gain_3.0,3.5.png')
plt.show()

plt.plot(limit,NN+poiss,label='NN')
plt.plot(limit,MLE+poiss,label='ETAS')
plt.legend()
plt.xlabel('time from start / hours')
plt.ylabel('log-likelihood')
plt.title('log-likelihood with increasing training size for M0_train = 2.5, M0_test = 3.0')
# plt.savefig('raw_loglik_3.0,3.5.png')