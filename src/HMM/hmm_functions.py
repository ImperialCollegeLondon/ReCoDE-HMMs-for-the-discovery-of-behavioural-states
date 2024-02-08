"""Module for functions to analyse HMM outputs."""

import numpy as np
import pandas as pd
from ethoscopy.misc.rle import rle


def hmm_pct_transition(state_array: np.array, total_states: np.array) -> pd.DataFrame:
    """Finds the proportion of instances of runs of each state per array/fly.

    Args:
        state_array:  1D numpy array produced from a HMM decoder.
        total_states: numerical array denoting the states in 'state_array'.
    """
    values, starts, lengths = rle(state_array)

    states_dict = {}

    def average(a: np.ndarray) -> float:
        """Finds the average of a binary array."""
        total = a.sum()
        count = len(a)
        av = total / count
        return av

    for i in total_states:
        states_dict[f"{i}"] = average(np.where(values == i, 1, 0))

    state_list = [states_dict]
    df = pd.DataFrame(state_list)

    return df


def hmm_mean_length(
    state_array: np.array, delta_t: float = 60, raw: bool = False
) -> pd.DataFrame:
    """Finds the mean length of each state run per array/fly.

    Returns:
        A dataframe with a state column containing the states id and a mean_length
        column.

    Args:
        state_array:  1D numpy array produced from a HMM decoder.
        delta_t : the time difference between each element of the array.
        raw : If true then length of all runs of each stae are returned, rather than the
        mean.
    """
    assert isinstance(raw, bool)
    delta_t_mins = delta_t / 60

    values, starts, lengths = rle(state_array)

    df = pd.DataFrame(data=zip(values, lengths), columns=["state", "length"])
    df["length_adjusted"] = df["length"].map(lambda lengths: lengths * delta_t_mins)

    if raw is True:
        return df
    else:
        gb_bout = df.groupby("state").agg(
            **{"mean_length": ("length_adjusted", "mean")}
        )
        gb_bout.reset_index(inplace=True)

        return gb_bout


def hmm_pct_state(
    state_array: np.array,
    time: np.array,
    total_states: list[int | float],
    avg_window: int = 30,
) -> pd.DataFrame:
    """Takes a window of n and finds percentage each state is present in that window.

    Returns:
        A dataframe with columns t, and states with their corresponding percentage per
        window.

    Args:
        state_array:  1D numpy array produced from a HMM decoder.
        time: 1D numpy array of the timestamps of state_array of equal length and same
        order.
        total_states: numerical array denoting the states in 'state_array'.
        avg_window: length of window given as elements of the array.
    """
    states_dict = {}

    def moving_average(a: np.array, n: int) -> np.array:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    for i in total_states:
        states_dict["state_{}".format(i)] = moving_average(
            np.where(state_array == i, 1, 0), n=avg_window
        )

    adjusted_time = time[avg_window - 1 :]

    df = pd.DataFrame.from_dict(states_dict)
    df.insert(0, "t", adjusted_time)

    return df
