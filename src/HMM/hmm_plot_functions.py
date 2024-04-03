"""Module for functions to analyse HMM outputs."""

import random
from functools import partial
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
from misc import bin_data, rle


def hmm_mean_length(state_array: list[int], delta_t: int=60, raw: bool=False) -> pd.DataFrame:
    """Finds the mean length of each state run per array/fly returns a dataframe with a state column containing the states id and a mean_length column.

    Args:
        state_array: 1D numpy array produced from a HMM decoder.
        delta_t: the time difference between each element of the array.
        raw: If true then length of all runs of each stae are returned, rather than the mean.

    Returns:
        If raw is False then a grouped dataframe is returned, if True the raw lengths as a dataframe
    """ # noqa: E501
    delta_t_mins = delta_t / 60

    V, S, L = rle(state_array)

    df = pd.DataFrame(data=zip(V, L), columns=["state", "length"])
    df["length_adjusted"] = df["length"].map(lambda le: le * delta_t_mins)

    if raw is True:
        return df
    else:
        gb_bout = df.groupby("state").agg(
            **{"mean_length": ("length_adjusted", "mean")}
        )
        gb_bout.reset_index(inplace=True)

        return gb_bout


def hmm_pct_state(state_array: np.array, time: np.array, total_states: list[int], avg_window: int=30) -> pd.DataFrame:
    """Finds the moving average of each state.

    Args:
        state_array: 1D numpy array produced from a HMM decoder
        time: 1D numpy array of the timestamps of state_array of equal length and same order
        total_states: numerical array denoting the states in 'state_array'
        avg_window: length of window given as elements of the array.

    Returns:
        A pandas dataframe with moving average applied to each state
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


def hmm_decode(d: pd.DataFrame, h: CategoricalHMM, b: int, var: str, fun: str, t: str="t") -> tuple[list[int],pd.Series]:
    """Decode a time series dataframe with its trained HMM.

    Args:
        d: Pandas dataframe with the time series and emission data
        h: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
        b: the time in seconds the data should be binned to
        var: the name (as a string) of the column with the emussion data
        fun: the function to apply to the aggregating column, i.e. "max", "mean", ...
        t: the name (as a string) of the column with the time series data

    Returns:
        Two nested lists: 1) the decoded states 2) the time series 
    """
    # bin the data to 60 second intervals with a selected column and function on that column
    bin_df = d.groupby("id", group_keys=False).apply(
        partial(bin_data, column=var, bin_column=t, function=fun, bin_secs=b)
    )

    gb = bin_df.groupby(bin_df.index)[f"{var}_{fun}"].apply(list)
    time_list = bin_df.groupby(bin_df.index)["t_bin"].apply(list)

    states_list = []

    for i, t, id in zip(gb, time_list, time_list.index):
        seq_o = np.array(i)
        seq = seq_o.reshape(-1, 1)
        logprob, states = h.decode(seq)

        states_list.append(states)

    return states_list, time_list


def plot_hmm_overtime(data: pd.DataFrame, hmm: CategoricalHMM, variable: str,
                    labels: list[str], colours: list[str], wrapped: bool=False, tbin: int=60, func: str="max", avg_window: int=30) -> None:
    """Creates a plot of the pct of all states. The y-axis-liklihood of being in a sleep state, x-axis-time in hours.

    Args:
        data: The data to be plotted
        hmm: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
        variable: the name (as a string) of the column with the emussion data
        labels: the names of the different states present in the hidden markov model
        colours: the name of the colours you wish to represent the different states, must be the same length as labels
        wrapped: if True the plot will be limited to a 24 hour day average. Default is False
        tbin: the time in seconds the data should be binned to, Default is 60
        func: the function to apply to the aggregating column, i.e. "max", "mean", ... . Default is "max"
        avg_window: the window in minutes you want the moving average to be applied to. Default is 30 mins

    Returns:
        None. But a matplotlib figure is plotted to screen.
    """
    data = data.copy(deep=True).reset_index()

    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError(
            "The number of labels do not match the number of states in the model"
        )

    states_list, time_list = hmm_decode(d = data, h = hmm, b = tbin, var = variable, fun = func)

    data = pd.DataFrame()
    for states, t in zip(states_list, time_list):
        tdf = hmm_pct_state(
            states, t, list(range(len(labels))), avg_window=int((avg_window * 60) / tbin)
        )
        data = pd.concat([data, tdf], ignore_index=True)

    if wrapped is True:
        data["t"] = data["t"].map(lambda t: t % (60 * 60 * 24))

    data["t"] = data["t"] / (60 * 60)
    int(12 * floor(data.t.min() / 12))
    int(12 * ceil(data.t.max() / 12))

    plt.figure(figsize=(10, 8))

    for c, (col, n) in enumerate(zip(colours, labels)):
        column = f"state_{c}"

        gb_df = data.groupby("t").agg(
            **{
                "mean": (column, "mean"),
                "SD": (column, "std"),
                "count": (column, "count"),
            }
        )

        gb_df["SE"] = (1.96 * gb_df["SD"]) / np.sqrt(gb_df["count"])
        gb_df["y_max"] = gb_df["mean"] + gb_df["SE"]
        gb_df["y_min"] = gb_df["mean"] - gb_df["SE"]
        gb_df = gb_df.reset_index()

        plt.plot(gb_df["t"], gb_df["mean"], color=col, label=n)
        plt.fill_between(
            gb_df["t"], gb_df["y_min"], gb_df["y_max"], color=col, alpha=0.25
        )

    plt.legend(loc="upper right")
    plt.ylabel("% time in state")
    plt.xlabel("time (hours)")
    plt.ylim((0, 1))

    plt.show()


def plot_hmm_quantify(data: pd.DataFrame, hmm: CategoricalHMM, variable: str, labels: list[str], colours: list[str], tbin: int=60, func: str="max") -> None:
    """Creates a boxplot of pct in each state.

    Args:
        data: The data to be plotted
        hmm: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
        variable: the name (as a string) of the column with the emussion data
        labels: the names of the different states present in the hidden markov model
        colours: the name of the colours you wish to represent the different states, must be the same length as labels
        tbin: the time in seconds the data should be binned to, Default is 60
        func: the function to apply to the aggregating column, i.e. "max", "mean", ... . Default is "max"

    Returns:
        None. But a seaborn figure is plotted to screen.
    """
    data = data.copy(deep=True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError(
            "The number of labels do not match the number of states in the model"
        )
    label_dict = {k: v for k, v in zip(list_states, labels)}

    rows = []
    for array in states_list:
        unique, counts = np.unique(array, return_counts=True)
        row = dict(zip(unique, counts))
        rows.append(row)
    counts_all = pd.DataFrame(rows)
    counts_all["sum"] = counts_all.sum(axis=1)
    counts_all = counts_all.iloc[:, list_states[0] : list_states[-1] + 1].div(
        counts_all["sum"], axis=0
    )
    counts_all.fillna(0, inplace=True)

    plt.figure(figsize=(10, 8))

    d = pd.melt(counts_all)
    d["State"] = d["variable"].map(label_dict)
    sns.set_style("whitegrid")
    ax = sns.swarmplot(
        data=d, x="State", y="value", hue="State", palette=colours, legend=False, size=3
    )
    ax = sns.boxplot(
        data=d,
        x="State",
        y="value",
        hue="State",
        palette=colours,
        showcaps=False,
        showfliers=False,
        whiskerprops={"linewidth": 0},
    )

    ax.set(ylabel="% of time in state")
    ax.set(ylim=(0, 1))

    means = d.groupby(["variable"])["value"].mean().round(3)
    vertical_offset = d["value"].mean() * 0.4

    for xtick in ax.get_xticks():
        ax.text(
            xtick,
            means[xtick] + vertical_offset,
            means[xtick],
            horizontalalignment="center",
            size="x-large",
            color="black",
            weight="bold",
        )


def plot_hmm_quantify_length(data: pd.DataFrame, hmm: CategoricalHMM, variable: str, labels: list[str], colours: list[str], tbin: int=60, func: str="max") -> None:
    """Creates a boxplot of the mean length of each state.

    Args:
        data: The data to be plotted
        hmm: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
        variable: the name (as a string) of the column with the emussion data
        labels: the names of the different states present in the hidden markov model
        colours: the name of the colours you wish to represent the different states, must be the same length as labels
        tbin: the time in seconds the data should be binned to, Default is 60
        func: the function to apply to the aggregating column, i.e. "max", "mean", ... . Default is "max"

    Returns:
        None. But a seaborn figure is plotted to screen.
        
    """
    data = data.copy(deep=True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError(
            "The number of labels do not match the number of states in the model"
        )
    label_dict = {k: v for k, v in zip(list_states, labels)}

    d = pd.DataFrame()
    for states in states_list:
        length = hmm_mean_length(states, delta_t=tbin) # type: ignore
        d = pd.concat([d, length], ignore_index=True)

    d["State"] = d["state"].map(label_dict)

    plt.figure(figsize=(10, 10))
    sns.set_style("whitegrid")

    ax = sns.swarmplot(
        data=d,
        x="State",
        y="mean_length",
        hue="State",
        palette=colours,
        legend=False,
        size=3,
    )
    ax = sns.boxplot(
        data=d,
        x="State",
        y="mean_length",
        hue="State",
        palette=colours,
        showcaps=False,
        showfliers=False,
        whiskerprops={"linewidth": 0},
    )

    ax.set(ylabel="mean length of state")
    ax.set(yscale="log")

    means = round(d.groupby(["state"])["mean_length"].mean())
    vertical_offset = d["mean_length"].median() * 0.25

    for xtick in ax.get_xticks():
        ax.text(
            xtick,
            means[xtick] + vertical_offset,
            means[xtick],
            horizontalalignment="center",
            size="x-large",
            color="black",
            weight="bold",
        )


def plot_hmm_raw(data: pd.DataFrame, hmm: CategoricalHMM, variable: str, colours: list[str], tbin: int=60, func: str="max") -> None:
    """Creates a plot showing the raw output from a hmm decoder.

    Args:
        data: The data to be plotted
        hmm: a trained categorical HMM from hmmlearn with the correct hidden states and emission states for your dataset
        variable: the name (as a string) of the column with the emussion data
        colours: the name of the colours you wish to represent the different states, must be the same length as labels
        tbin: the time in seconds the data should be binned to, Default is 60
        func: the function to apply to the aggregating column, i.e. "max", "mean", ... . Default is "max"

    Returns:
        None. But a matplotlib figure is plotted to screen.
    
    """
    data = data.copy(deep=True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    time_list = list(time_list)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(colours):
        raise RuntimeError(
            "The number of colours do not match the number of states in the model"
        )

    rand_ind = random.choice(list(range(0, len(states_list))))

    st = states_list[rand_ind]
    time = time_list[rand_ind]
    time = np.array(time) / 86400

    for c, i in enumerate(colours):
        if c == 0:
            col = np.where(st == c, colours[c], np.NaN)
        else:
            col = np.where(st == c, colours[c], col)

    plt.figure(figsize=(80, 10))

    plt.scatter(time, st, s=50 * 2, marker="o", c=col)
    plt.plot(
        time,
        st,
        marker="o",
        markersize=0,
        mfc="white",
        mec="white",
        c="black",
        lw=1,
        ls="-",
    )

    plt.xlabel("Time (days)")
    plt.ylabel("State")
