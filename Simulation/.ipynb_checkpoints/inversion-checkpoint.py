#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##############################################################################
# inversion of ETAS parameters
#
# as described by Mizrahi et al., 2021
# Leila Mizrahi, Shyam Nandan, Stefan Wiemer;
# The Effect of Declustering on the Size Distribution of Mainshocks.
# Seismological Research Letters 2021; doi: https://doi.org/10.1785/0220200231
##############################################################################

from scipy.optimize import minimize
from scipy.special import gamma as gamma_func, gammaln, gammaincc, exp1

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
import json
import os
import pprint

from functools import partial
import pyproj
from shapely.geometry import Polygon
import shapely.ops as ops

from mc_b_est import round_half_up, estimate_beta_tinti


def coppersmith(mag, typ):
    # result is in km

    # typ is one of the following:
    # 1: strike slip fault
    # 2: reverse fault
    # 3: normal fault
    # 4: oblique fault

    if typ == 1:
        # surface rupture length
        SRL = np.power(10, (0.74 * mag - 3.55))
        # subsurface rupture length
        SSRL = np.power(10, (0.62 * mag - 2.57))
        # rupture width
        RW = np.power(10, (0.27 * mag - 0.76))
        # rupture area
        RA = np.power(10, (0.9 * mag - 3.42))
        # average slip
        AD = np.power(10, (0.9 * mag - 6.32))

    elif typ == 2:
        # surface rupture length
        SRL = np.power(10, (0.63 * mag - 2.86))
        # subsurface rupture length
        SSRL = np.power(10, (0.58 * mag - 2.42))
        # rupture width
        RW = np.power(10, (0.41 * mag - 1.61))
        # rupture area
        RA = np.power(10, (0.98 * mag - 3.99))
        # average slip
        AD = np.power(10, (0.08 * mag - 0.74))

    elif typ == 3:
        # surface rupture length
        SRL = np.power(10, (0.5 * mag - 2.01))
        # subsurface rupture length
        SSRL = np.power(10, (0.5 * mag - 1.88))
        # rupture width
        RW = np.power(10, (0.35 * mag - 1.14))
        # rupture area
        RA = np.power(10, (0.82 * mag - 2.87))
        # average slip
        AD = np.power(10, (0.63 * mag - 4.45))

    elif typ == 4:
        # surface rupture length
        SRL = np.power(10, (0.69 * mag - 3.22))
        # subsurface rupture length
        SSRL = np.power(10, (0.59 * mag - 2.44))
        # rupture width
        RW = np.power(10, (0.32 * mag - 1.01))
        # rupture area
        RA = np.power(10, (0.91 * mag - 3.49))
        # average slip
        AD = np.power(10, (0.69 * mag - 4.80))

    return {
        'SRL': SRL,
        'SSRL': SSRL,
        'RW': RW,
        'RA': RA,
        'AD': AD
    }


def rectangle_surface(lat1, lat2, lon1, lon2):
    l = [[lon1, lat1],
         [lon1, lat2],
         [lon2, lat2],
         [lon2, lat1]]
    polygon = Polygon(l)

    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=polygon.bounds[1],
                lat2=polygon.bounds[3])),
        polygon)
    return geom_area.area / 1e6


def polygon_surface(polygon):
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=polygon.bounds[1],
                lat_2=polygon.bounds[3])),
        polygon)
    return geom_area.area / 1e6


def hav(theta):
    return np.square(np.sin(theta / 2))


def haversine(lat_rad_1, lat_rad_2, lon_rad_1, lon_rad_2, earth_radius=6.3781e3):
    # to calculate distance on a sphere
    d = 2 * earth_radius * np.arcsin(
        np.sqrt(
            hav(lat_rad_1 - lat_rad_2)
            + np.cos(lat_rad_1)
            * np.cos(lat_rad_2)
            * hav(lon_rad_1 - lon_rad_2)
        )
    )
    return d


def branching_ratio(theta, beta):
    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    d = np.power(10, log10_d)
    tau = np.power(10, log10_tau)

    eta = beta * k0 * np.pi * np.power(d, -rho) * np.power(tau, -omega) * np.exp(c / tau) * upper_gamma_ext(-omega,c / tau) / (rho * (-a + beta + gamma * rho))
    return eta


def to_days(timediff):
    return timediff / dt.timedelta(days=1)


def upper_gamma_ext(a, x):
    if a > 0:
        return gammaincc(a, x) * gamma_func(a)
    elif a == 0:
        return exp1(x)
    else:
        return (upper_gamma_ext(a + 1, x) - np.power(x, a)*np.exp(-x)) / a


def parameter_array2dict(theta):
    return dict(zip(
        ['log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho'],
        theta
    ))


def parameter_dict2array(parameters):
    order = ['log10_mu', 'log10_k0', 'a', 'log10_c', 'omega', 'log10_tau', 'log10_d', 'gamma', 'rho']
    return np.array([
        parameters[key] for key in order
    ])


def set_initial_values(ranges=None):
    if ranges is None:
        log10_mu_range = (-10, 0)
        log10_k0_range = (-4, 0)
        a_range = (0.01, 5.)
        log10_c_range = (-8, 0)
        omega_range = (-0.99, 1)
        log10_tau_range = (0.01, 5)
        log10_d_range = (-4, 1)
        gamma_range = (0.01, 5.)
        rho_range = (0.01, 5.)
    else:
        log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range = ranges

    log10_mu = np.random.uniform(*log10_mu_range)
    log10_k0 = np.random.uniform(*log10_k0_range)
    a = np.random.uniform(*a_range)
    log10_c = np.random.uniform(*log10_c_range)
    omega = np.random.uniform(*omega_range)
    log10_tau = np.random.uniform(*log10_tau_range)
    log10_d = np.random.uniform(*log10_d_range)
    gamma = np.random.uniform(*gamma_range)
    rho = np.random.uniform(*rho_range)

    return [
        log10_mu,
        log10_k0,
        a,
        log10_c,
        omega,
        log10_tau,
        log10_d,
        gamma,
        rho
    ]


def prepare_catalog(data, mc, coppersmith_multiplier, timewindow_start, timewindow_end, earth_radius,
                    delta_m=0):
    # precalculates distances in time and space between events that are potentially related to each other
    calc_start = dt.datetime.now()

    # only use data above completeness magnitude
    if delta_m > 0:
        data["magnitude"] = round_half_up(data["magnitude"] / delta_m) * delta_m
    relevant = data.query("magnitude >= @mc").copy()
    relevant.sort_values(by='time', inplace=True)

    # all entries can be sources, but targets only after timewindow start
    targets = relevant.query("time>=@timewindow_start").copy()

    # calculate some source stuff
    relevant["distance_range_squared"] = np.square(
        coppersmith(relevant["magnitude"], 4)["SSRL"] * coppersmith_multiplier
    )
    relevant["source_to_end_time_distance"] = to_days(timewindow_end - relevant["time"])
    relevant["pos_source_to_start_time_distance"] = to_days(timewindow_start - relevant["time"]).apply(
        lambda x: max(x, 0)
    )

    # translate target lat, lon to radians for spherical distance calculation
    targets['target_lat_rad'] = np.radians(targets['latitude'])
    targets['target_lon_rad'] = np.radians(targets['longitude'])
    targets["target_time"] = targets["time"]
    targets["target_id"] = targets.index
    targets["target_time"] = targets["time"]
    # columns that are needed later
    targets["source_id"] = 'i'
    targets["source_magnitude"] = 0.0
    targets["time_distance"] = 0.0
    targets["spatial_distance_squared"] = 0.0
    targets["source_to_end_time_distance"] = 0.0
    targets["pos_source_to_start_time_distance"] = 0.0

    targets = targets.sort_values(by="time")

    # define index and columns that are later going to be needed
    if pd.__version__ >= '0.24.0':
        index = pd.MultiIndex(
            levels=[[], []],
            names=["source_id", "target_id"],
            codes=[[], []]
        )
    else:
        index = pd.MultiIndex(
            levels=[[], []],
            names=["source_id", "target_id"],
            labels=[[], []]
        )

    columns = [
        "target_time",
        "source_magnitude",
        "spatial_distance_squared",
        "time_distance",
        "source_to_end_time_distance",
        "pos_source_to_start_time_distance"
    ]
    res_df = pd.DataFrame(index=index, columns=columns)

    df_list = []

    print('  number of sources:', len(relevant.index))
    print('  number of targets:', len(targets.index))
    for source in relevant.itertuples():
        stime = source.time

        # filter potential targets
        if source.time < timewindow_start:
            potential_targets = targets.copy()
        else:
            potential_targets = targets.query(
                "time>@stime"
            ).copy()
        targets = potential_targets.copy()

        if potential_targets.shape[0] == 0:
            continue

        # get values of source event
        slatrad = np.radians(source.latitude)
        slonrad = np.radians(source.longitude)
        drs = source.distance_range_squared

        # get source id and info of target events
        potential_targets["source_id"] = source.Index
        potential_targets["source_magnitude"] = source.magnitude

        # calculate space and time distance from source to target event
        potential_targets["time_distance"] = to_days(potential_targets["target_time"] - stime)

        potential_targets["spatial_distance_squared"] = np.square(
            haversine(
                slatrad,
                potential_targets['target_lat_rad'],
                slonrad,
                potential_targets['target_lon_rad'],
                earth_radius
            )
        )

        # filter for only small enough distances
        potential_targets.query("spatial_distance_squared <= @drs", inplace=True)

        # calculate time distance from source event to timewindow boundaries for integration later
        potential_targets["source_to_end_time_distance"] = source.source_to_end_time_distance
        potential_targets["pos_source_to_start_time_distance"] = source.pos_source_to_start_time_distance

        # append to resulting dataframe
        df_list.append(potential_targets)

    res_df = pd.concat(df_list)[["source_id", "target_id"] + columns].reset_index().set_index(
        ["source_id", "target_id"])

    print('    took', (dt.datetime.now() - calc_start), 'to prepare the distances\n')

    return res_df


def triggering_kernel(metrics, params):
    # given time distance in days and squared space distance in square km and magnitude of target event,
    # calculate the (not normalized) likelihood, that source event triggered target event

    time_distance, spatial_distance_squared, m = metrics
    theta, mc = params

    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    mu = np.power(10, log10_mu)
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    aftershock_number = k0 * np.exp(a * (m - mc))
    time_decay = np.exp(-time_distance / tau) / np.power((time_distance + c), (1 + omega))
    space_decay = 1 / np.power(
        (spatial_distance_squared + d * np.exp(gamma * (m - mc))),
        (1 + rho)
    )

    res = aftershock_number * time_decay * space_decay
    return res


def expectation_step(distances, target_events, source_events, params, verbose=False):
    calc_start = dt.datetime.now()
    theta, mc = params
    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    # print('I am doing the expectation step with parameters', theta)
    mu = np.power(10, log10_mu)
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    # calculate the triggering density values gij
    if verbose:
        print('    calculating gij')
    Pij_0 = distances.copy()
    Pij_0["gij"] = triggering_kernel(
        [
            Pij_0["time_distance"],
            Pij_0["spatial_distance_squared"],
            Pij_0["source_magnitude"]
        ],
        [theta, mc]
    )

    # calculate muj for each target. currently constant, could be improved
    target_events_0 = target_events.copy()
    target_events_0["mu"] = mu

    # calculate triggering probabilities Pij
    if verbose:
        print('    calculating Pij')
    Pij_0["tot_rates"] = 0
    Pij_0["tot_rates"] = Pij_0["tot_rates"].add(Pij_0["gij"].sum(level=1)).add(target_events_0["mu"])
    Pij_0["Pij"] = Pij_0["gij"].div(Pij_0["tot_rates"])

    # calculate probabilities of being triggered or background
    target_events_0["P_triggered"] = 0
    target_events_0["P_triggered"] = target_events_0["P_triggered"].add(Pij_0["Pij"].sum(level=1)).fillna(0)
    target_events_0["P_background"] = 1 - target_events_0["P_triggered"]

    # calculate expected number of background events
    if verbose:
        print('    calculating n_hat and l_hat\n')
    n_hat_0 = target_events_0["P_background"].sum()

    # calculate aftershocks per source event
    source_events_0 = source_events.copy()
    source_events_0["l_hat"] = Pij_0["Pij"].sum(level=0)

    print('    expectation step took ', dt.datetime.now() - calc_start)
    return Pij_0, target_events_0, source_events_0, n_hat_0


def expected_aftershocks(event, params, no_start=False, no_end=False):
    theta, mc = params

    log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    if no_start:
        if no_end:
            event_magnitude = event
        else:
            event_magnitude, event_time_to_end = event
    else:
        if no_end:
            event_magnitude, event_time_to_start = event
        else:
            event_magnitude, event_time_to_start, event_time_to_end = event

    number_factor = k0 * np.exp(a * (event_magnitude - mc))
    area_factor = np.pi * np.power(
        d * np.exp(gamma * (event_magnitude - mc)),
        -1 * rho
    ) / rho

    time_factor = np.exp(c/tau) * np.power(tau, -omega)  # * gamma_func(-omega)

    if no_start:
        time_fraction = upper_gamma_ext(-omega, c/tau)
    else:
        time_fraction = upper_gamma_ext(-omega, (event_time_to_start + c)/tau)
    if not no_end:
        time_fraction = time_fraction - upper_gamma_ext(-omega, (event_time_to_end + c)/tau)

    time_factor = time_factor * time_fraction

    return number_factor * area_factor * time_factor


def ll_aftershock_term(l_hat, g):
    mask = g != 0
    term = -1 * gammaln(l_hat + 1) - g
    term = term + l_hat * np.where(mask, np.log(g, where=mask), -300)
    return term


def neg_log_likelihood(theta, args):
    mc, n_hat, Pij, source_events, timewindow_length, area = args

    assert Pij.index.names == ("source_id", "target_id"), "Pij must have multiindex with names 'source_id', 'target_id'"
    assert source_events.index.name == "source_id", "source_events must have index with name 'source_id'"

    log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    source_events["G"] = expected_aftershocks(
        [
            source_events["source_magnitude"],
            source_events["pos_source_to_start_time_distance"],
            source_events["source_to_end_time_distance"]
        ],
        [theta, mc]
    )

    aftershock_term = ll_aftershock_term(
        source_events["l_hat"],
        source_events["G"],
    ).sum()

    # space time distribution term
    Pij["likelihood_term"] = (
        (omega * np.log(tau) - np.log(upper_gamma_ext(-omega, c/tau))
         + np.log(rho) + rho * np.log(
            d * np.exp(gamma * (Pij["source_magnitude"] - mc))
        ))
        - ((1 + rho) * np.log(
            Pij["spatial_distance_squared"] + (
                d * np.exp(gamma * (Pij["source_magnitude"] - mc))
            )
        ))
        - (1 + omega) * np.log(Pij["time_distance"] + c)
        - (Pij["time_distance"] + c) / tau
        - np.log(np.pi)

    )
    distribution_term = Pij["Pij"].mul(Pij["likelihood_term"]).sum()

    total = aftershock_term + distribution_term

    return -1 * total


def optimize_parameters(theta_0, ranges, args):
    start_calc = dt.datetime.now()

    mc, n_hat, Pij, source_events, timewindow_length, area = args
    log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range = ranges

    log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta_0
    mu = np.power(10, log10_mu)
    k0 = np.power(10, log10_k0)
    c = np.power(10, log10_c)
    tau = np.power(10, log10_tau)
    d = np.power(10, log10_d)

    # estimate mu independently and remove from parameters
    mu_hat = n_hat / (area * timewindow_length)
    theta_0_without_mu = log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho

    bounds = [
        log10_k0_range,
        a_range,
        log10_c_range,
        omega_range,
        log10_tau_range,
        log10_d_range,
        gamma_range,
        rho_range
    ]

    res = minimize(
        neg_log_likelihood,
        x0=theta_0_without_mu,
        bounds=bounds,
        args=args,
        tol=1e-12,
    )

    new_theta_without_mu = res.x
    new_theta = [np.log10(mu_hat), *new_theta_without_mu]

    print("    optimization step took ", dt.datetime.now() - start_calc)

    return np.array(new_theta)


def invert_etas_params(
        metadata,
        timewindow_end=None,
        globe=False,
        store_pij=False,
        store_results=True
):
    """
        Inverts ETAS parameters.
        metadata can be either a string (path to json file with stored metadata)
        or a dict. accepted & necessary keywords are:
            fn_catalog: filename of the catalog (absolute path or filename in current directory)
                        catalog is expected to be a csv file with the following columns:
                        id, latitude, longitude, time, magnitude
                        id needs to contain a unique identifier for each event
                        time contains datetime of event occurrence
                        see synthetic_catalog.csv for an example
            data_path: path where result data will be stored
            auxiliary_start: start date of the auxiliary catalog (str or datetime).
                             events of the auxiliary catalog act as sources, not as targets
            timewindow_start: start date of the primary catalog , end date of auxiliary catalog (str or datetime).
                             events of the primary catalog act as sources and as targets
            timewindow_end: end date of the primary catalog (str or datetime)
            mc: cutoff magnitude. catalog needs to be complete above mc
            delta_m: size of magnitude bins
            coppersmith_multiplier: events further apart from each other than
                                    coppersmith subsurface rupture length * this multiplier
                                    are considered to be uncorrelated (to reduce size of distance matrix)
            shape_coords: coordinates of the boundary of the region to consider
                          (list of lists, i.e. [[lon1, lat1], [lon2, lat2], [lon3, lat3]])

    """
    ####################
    # preparing metadata
    ####################
    print("PREPARING METADATA...\n")

    if isinstance(metadata, str):
        # if metadata is a filename, read the file (assuming it's json)
        with open(metadata, 'r') as f:
            parameters_dict = json.load(f)
    else:
        parameters_dict = metadata

    fn_catalog = parameters_dict["fn_catalog"]
    print("  using catalog: " + fn_catalog)

    data_path = parameters_dict["data_path"]
    if data_path == "":
        print("  Data will be stored in " + os.path.dirname(os.path.abspath(__file__)))
    else:
        print("  Data will be stored in " + data_path)

    auxiliary_start = pd.to_datetime(parameters_dict["auxiliary_start"])
    timewindow_start = pd.to_datetime(parameters_dict["timewindow_start"])
    timewindow_end = timewindow_end or pd.to_datetime(parameters_dict["timewindow_end"])
    print(
        "  Time Window: " + str(auxiliary_start)
        + " (aux) - " + str(timewindow_start) + " (start) - " + str(timewindow_end) + " (end)"
    )

    mc = parameters_dict["mc"]
    delta_m = parameters_dict["delta_m"]
    print("  mc is " + str(mc) + " and delta_m is " + str(delta_m))

    coppersmith_multiplier = parameters_dict["coppersmith_multiplier"]
    print("  coppersmith multiplier is " + str(coppersmith_multiplier))

    if globe:
        coordinates = []
    else:
        if type(parameters_dict["shape_coords"]) is str:
            coordinates = np.array(eval(parameters_dict["shape_coords"]))
        else:
            coordinates = np.array(parameters_dict["shape_coords"])
        pprint.pprint("  Coordinates of region: " + str(list(coordinates)))

    # defining some other stuff here..

    timewindow_length = to_days(timewindow_end - timewindow_start)

    fn_parameters = data_path + 'parameters.json'
    fn_ip = data_path + 'ind_and_bg_probs.csv'
    fn_src = data_path + 'sources.csv'
    fn_dist = data_path + 'distances.csv'
    fn_pij = data_path + 'pij.csv'

    # earth radius in km
    earth_radius = 6.3781e3

    if globe:
        area = earth_radius ** 2 * 4 * np.pi
    else:
        poly = Polygon(coordinates)
        area = polygon_surface(poly)
    print("  Region has " + str(area) + " square km")

    # ranges for parameters
    log10_mu_range = (-10, 0)
    log10_k0_range = (-4, 0)
    a_range = (0.01, 5.)
    log10_c_range = (-8, 0)
    omega_range = (-0.99, 1)
    log10_tau_range = (0.01, 5)
    log10_d_range = (-4, 3)
    gamma_range = (0.01, 5.)
    rho_range = (0.01, 5.)

    ranges = log10_mu_range, log10_k0_range, a_range, log10_c_range, omega_range, log10_tau_range, log10_d_range, gamma_range, rho_range

    # start inversion
    print("\n\nINITIALIZING\n")
    print("  reading data..\n")
    df_full = pd.read_csv(fn_catalog, index_col=0, parse_dates=["time"], dtype={"url": str, "alert": str})
    gdf = gpd.GeoDataFrame(
        df_full, geometry=gpd.points_from_xy(df_full.longitude, df_full.latitude))

    # filter for events in region of interest
    if not globe:
        df = gdf[gdf.intersects(poly)].copy()
        df.drop("geometry", axis=1, inplace=True)
    else:
        df = df_full

    print("  "+str(len(df)) + " out of " + str(len(df_full)) + " events lie within target region.")

    # filter for events above cutoff magnitude - delta_m/2
    if delta_m > 0:
        df["magnitude"] = round_half_up(df["magnitude"] / delta_m) * delta_m
    df.query("magnitude>=@mc-@delta_m/2", inplace=True)

    print("  "+str(len(df)) + " events are above cutoff magnitude")

    # filter for eventsin relevant timewindow
    df.query("time >= @ auxiliary_start and time < @ timewindow_end", inplace=True)

    print("  "+str(len(df)) + " events are within time window\n\n")

    print('  calculating distances..\n')

    distances = prepare_catalog(
        df,
        mc=mc - delta_m / 2,
        coppersmith_multiplier=coppersmith_multiplier,
        timewindow_start=timewindow_start,
        timewindow_end=timewindow_end,
        earth_radius=earth_radius,
        delta_m=delta_m
    )
    # distances.to_csv(fn_dist)

    print('  preparing source and target events..\n')

    target_events = df.copy()
    target_events.query("time > @ timewindow_start", inplace=True)
    target_events.index.name = "target_id"

    beta = estimate_beta_tinti(target_events["magnitude"], mc=mc, delta_m=delta_m)
    print("  beta of primary catalog is", beta)

    source_columns = [
        "source_magnitude",
        "source_to_end_time_distance",
        "pos_source_to_start_time_distance"
    ]

    source_events = pd.DataFrame(
        distances[
            source_columns
        ].groupby("source_id").first()
    )

    try:
        print('  using input initial values for theta\n')
        initial_values = parameter_dict2array(
            parameters_dict["theta_0"]
        )
    except KeyError:
        print('  randomly chosing initial values for theta\n')
        initial_values = set_initial_values()


    #################
    # start inversion
    #################
    print('\n\nSTART INVERSION!\n')

    diff_to_before = 100
    i = 0
    while diff_to_before >= 0.001:
        print('iteration ' + str(i) + '\n')

        if i == 0:
            parameters = initial_values

        print('  expectation\n')
        Pij, target_events, source_events, n_hat = expectation_step(
            distances=distances,
            target_events=target_events,
            source_events=source_events,
            params=[parameters, mc - delta_m / 2],
            verbose=True
        )
        print('      n_hat:', n_hat, '\n')

        print('  maximization\n')
        args = [mc - delta_m / 2, n_hat, Pij, source_events, timewindow_length, area]

        new_parameters = optimize_parameters(
            theta_0=parameters,
            args=args,
            ranges=ranges
        )
        print('    new parameters:\n')
        pprint.pprint(
            parameter_array2dict(new_parameters),
            indent=4
        )
        diff_to_before = np.sum(np.abs(parameters - new_parameters))
        print('\n    difference to previous:', diff_to_before)

        br = branching_ratio(parameters, beta)
        print('    branching ratio:', br, '\n')
        parameters = new_parameters
        i += 1

    print('stopping here. converged after', i, 'iterations.')
    print('  last expectation step\n')
    Pij, target_events, source_events, n_hat = expectation_step(
        distances=distances,
        target_events=target_events,
        source_events=source_events,
        params=[parameters, mc - delta_m / 2],
        verbose=True
    )
    print('      n_hat:', n_hat)
    if store_results:
        target_events.to_csv(fn_ip)
        source_events.to_csv(fn_src)

        all_info = {
            "auxiliary_start": str(auxiliary_start),
            "timewindow_start": str(timewindow_start),
            "timewindow_end": str(timewindow_end),
            "timewindow_length": timewindow_length,
            "mc": mc,
            "beta": beta,
            "n_target_events": len(target_events),
            "delta_m": delta_m,
            "shape_coords": str(list(coordinates)),
            "earth_radius": earth_radius,
            "area": area,
            "coppersmith_multiplier": coppersmith_multiplier,
            "log10_mu_range": log10_mu_range,
            "log10_k0_range": log10_k0_range,
            "a_range": a_range,
            "log10_c_range": log10_c_range,
            "omega_range": omega_range,
            "log10_tau_range": log10_tau_range,
            "log10_d_range": log10_d_range,
            "gamma_range": gamma_range,
            "rho_range": rho_range,
            "ranges": ranges,
            "fn": fn_catalog,
            "fn_dist": fn_dist,
            "fn_ip": fn_ip,
            "fn_src": fn_src,
            "calculation_date": str(dt.datetime.now()),
            "initial_values": str(parameter_array2dict(initial_values)),
            "final_parameters": str(parameter_array2dict(new_parameters)),
            "n_iterations": i
        }

        info_json = json.dumps(all_info)
        f = open(fn_parameters, "w")
        f.write(info_json)
        f.close()

    if store_pij:
        Pij.to_csv(fn_pij)

    return parameter_array2dict(new_parameters)
