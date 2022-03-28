from typing import Any, Dict, Tuple, Union, Optional, List

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon


PyCaretForecastingHorizonTypes = Union[List[int], int, np.ndarray, ForecastingHorizon]


def coverage_to_quantiles(coverage: Union[float, List[float]]) -> List[float]:
    """Converts the coverage values to quantile (alpha values).

    Parameters
    ----------
    coverage : Union[float, List[float]
        The coverage values to convert to quantile (alpha values). Can be a single
        value or a list of coverage values.

    Returns
    -------
    List[float]
        The converted alpha values
    """
    if not isinstance(coverage, list):
        coverage = [coverage]

    alphas = []
    for c in coverage:
        # compute quantiles corresponding to prediction interval coverage
        #  this uses symmetric predictive intervals
        alphas.extend([0.5 - 0.5 * float(c), 0.5 + 0.5 * float(c)])
        # Rounding due to numerical precision issue - i.e. we get 0.049999... instead of 0.05
        alphas = [round(alpha, 4) for alpha in alphas]
    return alphas


def reconcile_quantile_coverage(
    quantile: Union[float, List[float]], coverage: float
) -> Tuple[float, float, float]:
    """Reconciles the quantile (alpha values) and coverage values passed. Returns
    the quantiles (alpha values) to be used for point forecast along with lower and
    upper quantiles for intervals.

    Use Cases:
    (1) If quantile is a list, it is used as is. In this case, it must be of length
        3 corresponding to Point Quantile, Lower Quantile, Upper Quantile (order
        does not matter).
    (2) If both quantile and coverage are floating point numbers, then they are
        reconciled.

    Parameters
    ----------
    quantile : Union[float, List
        Quantiles (alpha values) to reconcile
    coverage : float
        Coverage to reconcile

    Returns
    -------
    Tuple[float, float, float]
        Point Quantile, Lower Quantile, Upper Quantile in that order

    Raises
    ------
    ValueError
        (1) If quantile is a list but not of length 3.
        (2) Point quantile does not lie strictly between the lower and upper
            quantiles derived from coverage.
    """
    if isinstance(quantile, list):
        if len(quantile) != 3:
            raise ValueError(
                "If quantile is a list, it must be of length 3, indicating "
                "lower, point and upper quantile for prediction."
            )

    if isinstance(quantile, list):
        quantile.sort()
        lower_quantile, point_quantile, upper_quantile = quantile
    else:
        point_quantile = quantile
        lower_quantile, upper_quantile = coverage_to_quantiles(coverage=coverage)
        if point_quantile >= upper_quantile or point_quantile <= lower_quantile:
            raise ValueError(
                "Point quantile must lie strictly between the lower and upper quantiles derived"
                f"from coverage. Provided:\n\tPoint Quantile = {point_quantile}"
                f"\n\tLower Quantile = {lower_quantile}\n\tUpper Quantile = {upper_quantile}"
            )

    return point_quantile, lower_quantile, upper_quantile


def get_predictions_with_intervals(
    forecaster,
    X: pd.DataFrame,
    quantile: List[float],
    fh=None,
    merge: bool = False,
    round: Optional[int] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Returns the predictions, lower and upper interval values for a
    forecaster. If the forecaster does not support prediction intervals,
    then NAN is returned for lower and upper intervals.

    Parameters
    ----------
    forecaster : sktime compatible forecaster
        Forecaster to be used to get the predictions
    X : pd.DataFrame
        Exogenous Variables
    quantile : List[float]
        Quantile (alpha values) for point forecast and prediction interval. Must
        be a list containing 3 values corresponding to the point forecast, lower
        and upper limits.
    merge : bool, default = False
        If True, returns a dataframe with 3 columns called
        ["y_pred", "lower", "upper"], else retruns 3 separate series.
    round : Optional[int], default = None
        If set to an integer value, returned values are rounded to as many digits
        If set to None, no rounding is performed.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
        Predictions, Lower and Upper Interval Values

    Raises
    ------
    ValueError
        If alpha is a list but not of length 3.
    """
    # TODO: This has been changed to take into account sktime 0.11.0 but has not
    # fully been tested. Continue testing once the following is resolved:
    # https://github.com/alan-turing-institute/sktime/issues/2317
    # https://github.com/alan-turing-institute/sktime/pull/2318

    # TODO: Also, Do we need to enforce a list of 3 values. If the forecaster
    # does not support prediction intervals, why should we force 3 values for quantiles?
    # Maybe yes, since this function is specifically to get "predictions with intervals".

    # Predict and get lower and upper intervals
    return_pred_int = forecaster.get_tag("capability:pred_int")
    reporting_msg = "\nPlease report on GitHub: https://github.com/pycaret/pycaret/issues/new/choose"

    if isinstance(quantile, Tuple):
        quantile = list(quantile)

    if len(quantile) != 3:
        raise ValueError(
            "alpha provided must be a list of length 3 indicating the lower, point "
            f"and upper quantiles.\nProvided values were: {quantile}. This condition "
            f"should not have occurred.{reporting_msg}"
        )

    quantile.sort()
    lower_quantile, point_quantile, upper_quantile = quantile
    if return_pred_int:
        if point_quantile == 0.5:
            # Some forecasters will not produce deterministic forecasts if we use
            # predict_quantiles for point forecasts. e.g. see sktime discussion:
            # https://github.com/alan-turing-institute/sktime/pull/2320
            # Hence if point quantile is 0.5, just use the regular predict method.
            y_pred = forecaster.predict(fh=fh, X=X)
            return_values = forecaster.predict_quantiles(
                fh=fh, X=X, alpha=[lower_quantile, upper_quantile]
            )
        else:
            # Only use predict_quantiles for point forecast if point forecast alpha
            # is not 0.5.
            return_values = forecaster.predict_quantiles(fh=fh, X=X, alpha=quantile)
            y_pred = pd.DataFrame(
                {"y_pred": return_values[("Quantiles", point_quantile)]}
            )

        lower = pd.DataFrame({"lower": return_values[("Quantiles", lower_quantile)]})
        upper = pd.DataFrame({"upper": return_values[("Quantiles", upper_quantile)]})
    else:
        if point_quantile == 0.5:
            return_values = forecaster.predict(fh=fh, X=X)
        else:
            raise ValueError(
                "Forecater does not support quantile values other than 0.5, but was "
                f"provided {point_quantile}. This condition should not have occurred."
                f"{reporting_msg}"
            )

        y_pred = pd.DataFrame({"y_pred": return_values})
        lower = pd.DataFrame({"lower": [np.nan] * len(y_pred)})
        upper = pd.DataFrame({"upper": [np.nan] * len(y_pred)})
        lower.index = y_pred.index
        upper.index = y_pred.index

    # PyCaret works on Period Index only when developing models. If user passes
    # DateTimeIndex, it gets converted to PeriodIndex. If the forecaster (such as
    # Prophet) does not support PeriodIndex, then a patched version is created
    # which can support a PeriodIndex input and returns a PeriodIndex prediction.
    # Hence, no casting of index needs to be done here.

    if round is not None:
        # Converting to float since rounding does not support int
        y_pred = y_pred.astype(float).round(round)
        lower = lower.astype(float).round(round)
        upper = upper.astype(float).round(round)

    if merge:
        results = pd.concat([y_pred, lower, upper], axis=1)
        return results
    else:
        return y_pred, lower, upper


def update_additional_scorer_kwargs(
    initial_kwargs: Dict[str, Any],
    y_train: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Dict[str, Any]:
    """Updates the initial kwargs with additional scorer kwargs
    NOTE: Initial kwargs are obtained from experiment, e.g. {"sp": 12} and
    are common to all folds.
    The additional kwargs such as y_train, lower, upper are specific to each
    fold and must be updated dynamically as such.

    Parameters
    ----------
    initial_kwargs : Dict[str, Any]
        Initial kwargs are obtained from experiment, e.g. {"sp": 12} and
        are common to all folds
    y_train : pd.Series
        Training Data. Used in metrics such as MASE
    lower : pd.Series
        Lower Limits of Predictions. Used in metrics such as INPI
    upper : pd.Series
        Upper Limits of Predictions. Used in metrics such as INPI

    Returns
    -------
    Dict[str, Any]
        Updated kwargs dictionary
    """
    additional_scorer_kwargs = initial_kwargs.copy()
    additional_scorer_kwargs.update(
        {"y_train": y_train, "lower": lower, "upper": upper}
    )
    return additional_scorer_kwargs
