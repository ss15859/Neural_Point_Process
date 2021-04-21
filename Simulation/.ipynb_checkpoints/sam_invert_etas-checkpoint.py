import numpy as np
import datetime as dt

from sam_inversion import invert_etas_params

if __name__ == '__main__':
    theta_0 = {
        'log10_mu': -5.8,
        'log10_k0': -2.6,
        'a': 1.8,
        'log10_c': -2.5,
        'omega': 1.2
    }

    inversion_meta = {
        "fn_catalog": "test_catalog.csv",
        "data_path": "",
        "auxiliary_start": dt.datetime(1969, 1, 1),
        "timewindow_start": dt.datetime(1969, 1, 1),
        "timewindow_end": dt.datetime(2180, 1, 1),
        "theta_0": theta_0,
        "mc": 3.0,
        "delta_m": 0.0,
        "coppersmith_multiplier": 100,
        "shape_coords": [list([a[1], a[0]]) for a in (np.load("california_shape.npy"))],
    }

    parameters = invert_etas_params(
        inversion_meta,
    )

    print(parameters)
