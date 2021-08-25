import pandas as pd 
import numpy as np
import csv
import os
import sys

M0=0.3

for Mcut in [2.0,2.5,3.0]:
	for timeupto in np.linspace(600,4200,7):

		timeupto = int(timeupto)

		filename = '~/PhD/Amatrice_tests/paramsMcut:'+str(Mcut)+'00000_timefrom:0_timeupto:' + str(timeupto)+ '.csv'

		params = pd.read_csv(filename)    
		    
		MLE = {'mu': params.x[0], 'k0': params.x[1],'a':params.x[2],'c':params.x[3],'omega':params.x[4],'M0':M0,'beta': 1.5337497886680227}

		newfilename = 'paramsMcut-'+str(Mcut)+'_timeupto:' + str(timeupto)+ '.csv'

		with open(newfilename, "w") as csv_file:  
		        writer = csv.writer(csv_file)
		        for key, value in MLE.items():
		           writer.writerow([key, value])


