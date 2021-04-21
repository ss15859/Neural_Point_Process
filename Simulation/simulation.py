#!/usr/bin/env python
# coding: utf-8

##############################################################################
# simulation of earthquake catalogs using ETAS
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
##############################################################################

import pandas as pd
import numpy as np
import datetime as dt
import geopandas as gpd
from scipy.special import gammaincc, gammainccinv, gamma

from inversion import to_days, branching_ratio, \
    haversine, expected_aftershocks
from mc_b_est import simulate_magnitudes


from shapely.geometry import Polygon



def parameter_dict2array(parameters):
    order = ['mu', 'k0', 'a', 'c', 'omega', 'tau']
    return np.array([
        parameters[key] for key in order
    ])


def simulate_aftershock_time(c, omega, tau, size=1):
    # time delay in days

    y = np.random.uniform(size=size)

    return np.power(np.power(c,omega-1)/(1-y),1/(omega-1))-c

def productivity(m,k0,a,M0):
    return k0*np.exp(a*(m-M0))


def generate_background_events(timewindow_start, timewindow_end,
                               parameters, beta, mc):
    
    from inversion import polygon_surface, to_days

   
    timewindow_length = to_days(timewindow_end - timewindow_start)


    # number of background events
    expected_n_background = parameters["mu"]  * timewindow_length
    n_background = np.random.poisson(lam=expected_n_background)



    # define dataframe with background events
    catalog = pd.DataFrame(None, columns=["time", "magnitude", "parent", "generation"])



    # generate time, magnitude
    catalog["time"] = [
        timewindow_start
        + dt.timedelta(days=d) for d in np.random.uniform(0, timewindow_length, size=n_background)
    ]

    catalog["magnitude"] = simulate_magnitudes(n_background, beta=beta, mc=mc)

    # info about origin of event
    catalog["generation"] = 0
    catalog["parent"] = 0
    catalog["is_background"] = True

    # reindexing
    catalog = catalog.sort_values(by="time").reset_index(drop=True)
    catalog.index += 1
    catalog["gen_0_parent"] = catalog.index



    # simulate number of aftershocks
    catalog["expected_n_aftershocks"] = productivity(
        catalog["magnitude"],parameters["k0"],parameters["a"],mc)
    catalog["n_aftershocks"] = np.random.poisson(lam=catalog["expected_n_aftershocks"])

    return catalog


def generate_aftershocks(sources, generation, parameters, beta, mc, timewindow_end, timewindow_length):
    

    all_aftershocks = []

    # random timedeltas for all aftershocks
    total_n_aftershocks = sources["n_aftershocks"].sum()

    all_deltas = simulate_aftershock_time(
        c=parameters["c"],
        omega=parameters["omega"],
        tau=parameters["tau"],
        size=total_n_aftershocks
    )

    print(len(all_deltas))
    aftershocks = sources.loc[sources.index.repeat(sources.n_aftershocks)]

    keep_columns = ["time", "magnitude"]
    aftershocks["parent"] = aftershocks.index

    for col in keep_columns:
        aftershocks["parent_" + col] = aftershocks[col]

    # time of aftershock
    aftershocks = aftershocks[[col for col in aftershocks.columns if "parent" in col]].reset_index(drop=True)
    aftershocks["time_delta"] = all_deltas
    aftershocks.query("time_delta <= @ timewindow_length", inplace=True)

    aftershocks["time"] = aftershocks["parent_time"] + pd.to_timedelta(aftershocks["time_delta"], unit='d')
    aftershocks.query("time <= @ timewindow_end", inplace=True)

    

    as_cols = [
        "parent",
        "gen_0_parent",
        "time"
    ]


    aadf = aftershocks[as_cols].reset_index(drop=True)

    # magnitudes
    n_total_aftershocks = len(aadf.index)
    aadf["magnitude"] = simulate_magnitudes(n_total_aftershocks, beta=beta, mc=mc)

    # info about generation and being background
    aadf["generation"] = generation + 1
    aadf["is_background"] = False

    # info for next generation
    aadf["expected_n_aftershocks"] = productivity(
        aadf["magnitude"],parameters["k0"],parameters["a"],mc)
    aadf["n_aftershocks"] = np.random.poisson(lam=aadf["expected_n_aftershocks"])

    return aadf


def prepare_auxiliary_catalog(auxiliary_catalog, parameters, mc):
    theta_without_mu = parameters["k0"], parameters["a"], parameters["c"], parameters["omega"], \
                       parameters["tau"]

    catalog = auxiliary_catalog.copy()

    catalog.loc[:, "generation"] = 0
    catalog.loc[:, "parent"] = 0
    catalog.loc[:, "is_background"] = False

    # reindexing
    catalog["evt_id"] = catalog.index.values
    catalog = catalog.sort_values(by="time").reset_index(drop=True)
    catalog.index += 1
    catalog["gen_0_parent"] = catalog.index

    # simulate number of aftershocks
    catalog["expected_n_aftershocks"] = productivity(
        catalog["magnitude"],parameters["k0"],parameters["a"],mc)

    catalog["n_aftershocks"] = catalog["expected_n_aftershocks"].apply(
        np.random.poisson,
        # axis = 1
    )

    return catalog


def generate_catalog(
        timewindow_start, timewindow_end,
        parameters, mc, beta_main,gaussian_scale=None
):
    """
    Simulates an earthquake catalog.
        polygon: lon lat coordinates in which catalog is generated
        timewindow_start: datetime of simulation start
        timewindow_end: datetime of simulation end
        parameters: as estimated in the ETAS EM inversion
        mc: completeness magnitude. if delta_m > 0, magnitudes are simulated above mc-delta_m/2
        beta_main: beta used to generate background event magnitudes,
        beta_aftershock: beta used to generate aftershock magnitudes. if none, beta_main is used
        delta_m: bin size of magnitudes

        optional: use coordinates and independence probabilities
        of observed events to simulate locations of background events
        background_lats: list of latitudes
        background_lons: list of longitudes
        background_probs: list of independence probabilities
            these three lists are assumed to be sorted
            such that corresponding entries belong to the same event
        gaussian_scale: sigma to be used when background loations are generated
    """



    # generate background events
    print("generating background events..")
    catalog = generate_background_events(
        timewindow_start, timewindow_end, parameters, beta=beta_main, mc=mc)

    theta = parameters["mu"], parameters["k0"], parameters["a"], parameters["c"], parameters["omega"], \
            parameters["tau"]


    print('  number of background events:', len(catalog.index))

    generation = 0
    timewindow_length = to_days(timewindow_end - timewindow_start)

    while True:
        print('\n\nsimulating aftershocks of generation', generation, '..')
        sources = catalog.query("generation == @generation and n_aftershocks > 0").copy()

        # if no aftershocks are produced by events of this generation, stop
        print('  number of events with aftershocks:', len(sources.index))

        if len(sources.index) == 0:
            break

        # an array with all aftershocks. to be appended to the catalog
        aftershocks = generate_aftershocks(
            sources, generation, parameters, beta_main, mc,
            timewindow_end=timewindow_end, timewindow_length=timewindow_length,
        )

        aftershocks.index += catalog.index.max() + 1

        print('  number of generated aftershocks:', len(aftershocks.index))

        catalog = catalog.append(aftershocks, ignore_index=False, sort=True)

        generation = generation + 1

    print('\n\ntotal events simulated!:', len(catalog))


    return catalog
