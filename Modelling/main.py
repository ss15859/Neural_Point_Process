import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt

from NeuralPP import NPP

from ETAS import marked_ETAS_intensity, marked_likelihood, H

import ETAS


# read in data and sort by time


data = pd.read_csv('/home/ss15859/Documents/Mini_Project/Neural_Point_Process/Simulation/my_synthetic_catalog.csv',index_col=0)
data = data.sort_values('time')


#format time in days

dates = list(data['time'])
dates_list = (np.array([dt.datetime.strptime(date[:-3], "%Y-%m-%d %H:%M:%S.%f") for date in dates]))
times = (dates_list-dates_list[0])/ dt.timedelta(days=1)
times=times.astype('float64')


#test-train split

n_test = 3000
n_train = data.shape[0]-n_test

# n_train  = math.floor(0.8*data.shape[0])


M_train = np.array(data['magnitude'][:n_train])
M_test = np.array(data['magnitude'][n_train:])
T_train = np.array(times[:n_train])
T_test  = np.array(times[n_train:])

# read in and set parameters

params = pd.read_csv('/home/ss15859/Documents/Mini_Project/Neural_Point_Process/Simulation/params.csv')

tau=10**float(params["log10_tau"])
c= 10**float(params["log10_c"])
w = float(params["omega"])
k0 = 10**float(params["log10_k0"])
alpha = float(params["a"])
M0 = float(params["M0"])
d = 10**float(params["log10_d"])
rho = float(params["rho"])
ground = 10**float(params["log10_mu"])
gama = float(params["gamma"])
beta = float(params["beta"])
area = float(params["area"])

print('Calculating true lambda')

# calculate true intensity function

lam = marked_ETAS_intensity(T_test,M_test,ground,area,k0,alpha,M0,d,gama,rho,c,tau,w)

# shift to the right by truncation parameter for pfuture lotting

time_step=20
timesplot=T_test[time_step+1:]
lam=lam[time_step+1:]

# loop over increasing size of training set and calculate relative RMSE and mean log-likelihood


RMSE=np.zeros(16)

LL=np.zeros(16)

ln = np.linspace(np.log(10000),np.log(n_train),16)

for i in range(16):
    
    n=math.floor(np.exp(ln[i]))
    
    print('train size:  ' + str(n))

# define, train and predict with network    
    
    npp1 = NPP(time_step=time_step,size_rnn=64,size_nn=64,size_layer_chfn=2,size_layer_cmfn=2).set_train_data(T_train[:n],M_train[:n]).set_model().compile(lr=1e-3).fit_eval(epochs=30,batch_size=256).set_test_data(T_test,M_test).predict_eval()


# plot comparison of intensity functions around the largest magnitude event

    j = M_test.argmax()-time_step-1
    index=range(j,j+15)
    plt.plot(timesplot[index],npp1.lam[index],label='predicted')
    plt.plot(timesplot[index],lam[index],label='true')
    plt.legend()
    plt.show()


# calculate RMSE
    
    RMSE[i]=np.sqrt((1/lam.shape[0])*(((npp1.lam[:,0]-lam)/lam)**2).sum())
    print('RMSE:   '+str(RMSE[i]))


    LL[i]=npp1.LL.mean()
    
    
# plot RMSE with training size
    
plt.plot(ln,RMSE)
plt.xlabel('log n')
plt.ylabel('relative RMSE')
plt.title('relative RMSE with training size')
plt.show()


# calculate true mean log-likelihood and plot against training size

TLL = marked_likelihood(T_test[time_step+1:]-T_test[time_step+1],M_test[time_step+1:],T_test[-1]-T_test[time_step+1],ground,area,k0,alpha,M0,d,gama,rho,c,tau,w)/len(T_test[time_step+1:])


plt.plot(ln,LL)
plt.hlines(TLL,9,14,linestyles='dashed',colors='r',label='Calculated True MLL')
plt.xlabel('log n')
plt.ylabel('Mean LL')
plt.title('Mean Log-likelihood with training size')
plt.show()



# for largest training size plot relative absolute error over time with magnitudes

fig, (ax1, ax2) = plt.subplots(2,sharex=True)
fig.suptitle('relative absolute error in relation to magnitude of event')
ax1.plot(abs(npp1.lam[:,0]-lam)/lam,label="RMSE")
ax2.scatter(np.where(M_test>5.6),M_test[M_test>5.6],marker="x",label='Mag > 5.6',c="r")
# ax2.legend()
plt.xlabel("time")
plt.show()






    

    

