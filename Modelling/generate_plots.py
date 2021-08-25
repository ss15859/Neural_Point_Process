import pandas as pd 
import numpy as np
import csv
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


#################################################################### temp likelihood plot
Mcut = float(sys.argv[1])

train_times = np.linspace(600,4200,7)

ETAStemp = np.zeros(7)
NNtemp = np.zeros(7)

ETASmark = np.zeros(7)
NNmark = np.zeros(7)

i=0
for timeupto in train_times:

	timeupto=int(timeupto)

	filename  = 'data/resultsMcut-'+str(Mcut)+'_timeupto:' + str(timeupto)+ '.csv'
	D = pd.read_csv(filename)
	NNtemp[i] = D.NNLLtemp[0]
	ETAStemp[i] = D.ETASLLtemp[0]
	NNmark[i] = D.NNLLmark[0]
	ETASmark[i] = D.ETASLLmark[0]

	i+=1



d= {'Trained up to /hours':train_times, 'ETAStemp':ETAStemp,'NNtemp':NNtemp,'ETASmark':ETASmark,'NNmark':NNmark}
df = pd.DataFrame(d)

# plt,axs = sns.lineplot(x='Trained up to /hours', y='value', hue='variable', 
#              data=pd.melt(df, ['Trained up to /hours']))


sns.lineplot(data = df,x='Trained up to /hours', y='NNtemp',label='NN')
sns.lineplot(data = df,x='Trained up to /hours', y='ETAStemp',label = 'ETAS')
plt.ylabel('Mean Log-Likelihood Gain')
plt.title('Temporal Mean Log Likelihood Gain, Mcut = '+str(Mcut))
plt.show()

sns.lineplot(data = df,x='Trained up to /hours', y='NNmark',label='NN')
sns.lineplot(data = df,x='Trained up to /hours', y='ETASmark',label = 'ETAS')
plt.ylabel('Mean Log-Likelihood of Magnitude')
plt.title('Magnitude Mean Log Likelihood, Mcut = '+str(Mcut))
plt.show()