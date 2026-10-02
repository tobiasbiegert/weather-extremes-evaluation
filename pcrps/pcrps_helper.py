import numpy as np
import time
import xarray as xr
from isodisreg import idr

def get_points(predictions):
    return np.array(predictions.points)

def get_cdf(predictions):
    return np.array(predictions.ecdf)

def modify_points(points):
    return np.hstack([points[0], np.diff(points)])

def _pcrps(y, p, w, x):
    """
    Compute PCRPS for a single observation and a single IDR predictive distribution.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like
        Support points of the predictive distribution.

    Returns
    -------
    float
        PCRPS value for this observation and IDR predictive distribution.
    """
    return 2*np.sum(w*(np.array((y<=x))-p+0.5*w)*np.array(x-y))

def _tw_pcrps_upper(y, p, w, x, t):
    """
    Compute upper-tail threshold-weighted PCRPS (twPCRPS) for a single observation and
    a single IDR predictive distribution at given threshold levels.

    Parameters
    ----------
    y : float
        Observed value.
    p : array-like, shape (n_points,)
        CDF values at support points x.
    w : array-like
        Bin weights derived from the CDF.
    x : array-like, shape (n_points,)
        Support points of the predictive distribution.
    t : array-like
        One or more thresholds.

    Returns
    -------
    np.ndarray, shape (n_thresholds,)
        twPCRPS values for each threshold in `t`.
    """
    t = np.atleast_1d(t)
    x_clip = np.maximum(x[None, :], t[:, None])
    y_clip = np.maximum(y, t)

    w2 = w[None, :]
    p2 = p[None, :]
    indicator = (y_clip[:, None] <= x_clip).astype(float)
    terms = w2 * (indicator - p2 + 0.5 * w2) * (x_clip - y_clip[:, None])

    return 2 * np.sum(terms, axis=1)

def compute_easyuq(preds, obs, var, lead_time, level=None, compute_tw=False, thresholds=None):
    """
    Fit EasyUQ at each latitude/longitude point and compute PCRPS and optionally twPCRPS.

    Parameters
    ----------
    preds : xarray.DataArray
        Data array of model outputs for the variable of interest.
    obs : xarray.DataArray
        Observed data array for the variable of interest.
    var : str
        Name of the variable.
    lead_time : numpy.timedelta64
        Prediction lead time.
    level : int, optional
        Vertical level, if applicable.
    compute_tw : bool, optional
        If True, also compute threshold-weighted PCRPS.
    thresholds : xarray.DataArray, optional
        Thresholds at which twPCRPS is evaluated.

    Returns
    -------
    xarray.Dataset
        Dataset containing PCRPS and, if requested, twPCRPS for every grid point and
        initialization time, together with the corresponding computation times.
    """
    longitudes = preds.longitude.values
    latitudes = preds.latitude.values

    n_longitudes = len(longitudes)
    n_latitudes = len(latitudes)
    n_times = preds.sizes["time"]
    n_lead_times = preds.sizes["prediction_timedelta"]

    # Each Beam chunk must contain exactly one lead time.
    if n_lead_times != 1:
        raise ValueError("compute_easyuq expects exactly one prediction_timedelta per chunk.")

    tasks = ["pcrps_time"]

    if compute_tw:
        threshold_values = thresholds.threshold.values
        tasks.append("tw_pcrps_time")

    # Allocate the complete output arrays for this spatial chunk once.
    pcrps_values = np.empty(
        (n_longitudes, n_latitudes, n_lead_times, n_times),
        dtype=np.float32,
    )

    timing_values = np.empty(
        (n_longitudes, n_latitudes, n_lead_times, len(tasks)),
        dtype=np.float32,
    )

    if compute_tw:
        tw_pcrps_values = np.empty(
            (n_longitudes, n_latitudes, n_lead_times, n_times, len(threshold_values)),
            dtype=np.float32,
        )

    # Iterate over every grid point in the chunk.
    for latitude_index, latitude in enumerate(latitudes):
        for longitude_index, longitude in enumerate(longitudes):
            preds_point = preds.sel(
                latitude=latitude,
                longitude=longitude,
                prediction_timedelta=lead_time,
            )
            preds_point_df = preds_point.to_dataframe()[[var]]

            valid_time = preds_point.time + lead_time

            y = obs.sel(
                latitude=latitude,
                longitude=longitude,
                time=valid_time,
            ).values

            start_time_pcrps = time.time()

            fitted_idr = idr(y, preds_point_df)
            easyuq_preds_point = fitted_idr.predict(preds_point_df, digits=8)
            predictions = easyuq_preds_point.predictions

            if y.ndim > 1:
                raise ValueError("obs must be a 1-D array.")
            if np.any(np.isnan(y)):
                raise ValueError("obs contains NaN values.")
            if y.size != 1 and len(y) != len(predictions):
                raise ValueError("obs must have length 1 or the same length as predictions.")

            x = list(map(get_points, predictions))
            p = list(map(get_cdf, predictions))
            w = list(map(modify_points, p))

            # PCRPS
            pcrps = np.float32(list(map(_pcrps, y, p, w, x)))
            pcrps_time = np.float32(time.time() - start_time_pcrps)

            pcrps_values[longitude_index, latitude_index, 0, :] = pcrps
            timing_values[longitude_index, latitude_index, 0, 0] = pcrps_time

            # twPCRPS
            if compute_tw:
                t = thresholds.sel(
                    latitude=latitude,
                    longitude=longitude,
                    time=valid_time,
                ).values

                start_time_tw_pcrps = time.time()

                tw_pcrps = np.empty((n_times, len(threshold_values)), dtype=np.float32)

                for j, threshold_name in enumerate(threshold_values):
                    t_i = t[:, j]

                    # Transform forecasts and outcomes.
                    if threshold_name == "monthly_min":
                        x_tilde = t_i - preds_point_df[var]
                        y_tilde = t_i - y
                    else:
                        x_tilde = preds_point_df[var] - t_i
                        y_tilde = y - t_i

                    transformed_preds_df = x_tilde.to_frame(name=var)

                    # Fit EasyUQ
                    fitted_idr_tw = idr(y_tilde, transformed_preds_df)
                    easyuq_preds_tw = fitted_idr_tw.predict(transformed_preds_df, digits=8)
                    predictions_tw = easyuq_preds_tw.predictions

                    x_tw = list(map(get_points, predictions_tw))
                    p_tw = list(map(get_cdf, predictions_tw))
                    w_tw = list(map(modify_points, p_tw))

                    # Evaluate the transformed distributions at threshold zero.
                    tw_pcrps[:, j] = np.float32(list(map(_tw_pcrps_upper, y_tilde, p_tw, w_tw, x_tw, np.zeros(n_times))))[:, 0]

                tw_pcrps_time = np.float32(time.time() - start_time_tw_pcrps)

                tw_pcrps_values[longitude_index, latitude_index, 0, :, :] = tw_pcrps
                timing_values[
                    longitude_index,
                    latitude_index,
                    0,
                    tasks.index("tw_pcrps_time"),
                ] = tw_pcrps_time

    coordinates = {
        "longitude": longitudes,
        "latitude": latitudes,
        "prediction_timedelta": preds.prediction_timedelta.values,
        "time": preds.time.values,
        "task": tasks,
    }

    score_dimensions = (
        "longitude",
        "latitude",
        "prediction_timedelta",
        "time",
    )
    timing_dimensions = (
        "longitude",
        "latitude",
        "prediction_timedelta",
        "task",
    )

    if level is not None:
        coordinates["level"] = [level]

        score_dimensions = (
            "longitude",
            "latitude",
            "prediction_timedelta",
            "level",
            "time",
        )
        timing_dimensions = (
            "longitude",
            "latitude",
            "prediction_timedelta",
            "level",
            "task",
        )

        pcrps_values = pcrps_values[:, :, :, None, :]
        timing_values = timing_values[:, :, :, None, :]

        if compute_tw:
            tw_pcrps_values = tw_pcrps_values[:, :, :, None, :, :]

    data_variables = {
        f"{var}_pcrps": (score_dimensions, pcrps_values),
        f"{var}_time": (timing_dimensions, timing_values),
    }

    if compute_tw:
        coordinates["threshold"] = threshold_values
        tw_dimensions = (*score_dimensions, "threshold")
        data_variables[f"{var}_tw_pcrps"] = (tw_dimensions, tw_pcrps_values)

    return xr.Dataset(data_variables, coords=coordinates)