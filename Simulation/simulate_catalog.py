import csv
import numpy as np
import datetime as dt


from simulation import generate_catalog

if __name__ == '__main__':
    fn_store = 'my_synthetic_catalog.csv'
    burn_start = dt.datetime(1969, 1, 1)
    primary_start = dt.datetime(1969, 1, 1)
    end = dt.datetime(2120, 1, 1)
    test_end=dt.datetime(1994,1,1)

    mc = 3.0
    beta = 2.4

    parameters = {
    'mu': 1.6,
    'k0': 0.2,
    'a': 1.5,
    'c': 0.5,
    'omega': 1.5,
    'tau': np.power(10,3.99),
    'M0':mc,
    'beta':beta
    }

    # np.random.seed(777)

    synthetic = generate_catalog(
        timewindow_start=burn_start,
        timewindow_end=end,
        parameters=parameters,
        mc=mc,
        beta_main=beta
    )
    
    synthetic = synthetic.sort_values('time')

#     synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    synthetic.index.name = 'id'
    print("store catalog..")
    synthetic[["time", "magnitude"]].to_csv(fn_store)
    print("\nDONE!")
    
    
    with open('params.csv', 'w') as f:  
        w = csv.DictWriter(f, parameters.keys())
        w.writeheader()
        w.writerow(parameters)

        
        
        
    print('Generating test catalog')
    
    test = generate_catalog(
        timewindow_start=burn_start,
        timewindow_end=test_end,
        parameters=parameters,
        mc=mc,
        beta_main=beta
    )
    
    test = test.sort_values('time')

#     synthetic.magnitude = round_half_up(synthetic.magnitude, 1)
    test.index.name = 'id'
    print("store catalog..")
    test[["time", "magnitude"]].to_csv("test_catalog.csv")
    print("\nDONE!")