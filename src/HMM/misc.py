"""Module for functions to analyse HMM outputs."""
from math import floor
from typing import Callable

import numpy as np
import pandas as pd
import tabulate  # type: ignore


def rle(x: np.array) -> tuple[np.array, np.array, np.array]:
    """Find runs of consecutive items in an array.

    Args:
        x: ID numpy array

    Returns:
        Three arrays containg the run values, the start indices of the runs, and the
        lengths of the runs.
    """
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def bin_data(
    data: pd.DataFrame, column: str, bin_column: str, function: str, bin_secs: int
) -> pd.DataFrame:
    """A method to bin all data poits to a larger time bin, then summarise a column."""
    index_name = data["id"].iloc[0]

    data[bin_column] = data[bin_column].map(lambda t: bin_secs * floor(t / bin_secs))
    output_parse_name = f"{column}_{function}"  # create new column name

    bout_gb = data.groupby(bin_column).agg(**{output_parse_name: (column, function)})

    bin_parse_name = f"{bin_column}_bin"
    bout_gb.rename_axis(bin_parse_name, inplace=True)
    bout_gb.reset_index(level=0, inplace=True)
    old_index = pd.Index([index_name] * len(bout_gb.index), name="id")
    bout_gb.set_index(old_index, inplace=True)

    return bout_gb


def bootstrap(
    data: np.ndarray, n: int = 1000, func: Callable[[np.ndarray], float] = np.mean
) -> tuple[float, float]:
    """Generates bootstrap samples.

    Generate n bootstrap samples, evaluating `func` at each resampling. `bootstrap`
    returns a function, which can be called to obtain confidence intervals of interest

    Args:
        data: numpy array
        n: number of iterations of the simulation
        func: function to find average of all simulation outputs
    """
    simulations = list()
    sample_size = len(data)
    # xbar_init = np.mean(data)

    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    def ci(p: float) -> tuple[float, float]:
        """Return 2-sided symmetric confidence interval specified by p."""
        u_pval = (1 + p) / 2.0
        l_pval = 1 - u_pval
        l_indx = int(np.floor(n * l_pval))
        u_indx = int(np.floor(n * u_pval))
        return (simulations[l_indx], simulations[u_indx])

    return ci(0.95)


def _hmm_table(
    start_prob, trans_prob, emission_prob, state_names, observable_names
) -> None:
    """Prints a table of the probabilities from a hmmlearn MultinomialHMM object."""
    df_s = pd.DataFrame(start_prob)
    df_s = df_s.T
    df_s.columns = state_names
    print("Starting probabilty table: ")
    print(tabulate(df_s, headers="keys", tablefmt="github") + "\n")
    print("Transition probabilty table: ")
    df_t = pd.DataFrame(trans_prob, index=state_names, columns=state_names)
    print(tabulate(df_t, headers="keys", tablefmt="github") + "\n")
    print("Emission probabilty table: ")
    df_e = pd.DataFrame(emission_prob, index=state_names, columns=observable_names)
    print(tabulate(df_e, headers="keys", tablefmt="github") + "\n")


def hmm_display(hmm, states, observables) -> None:
    """Prints the transion probabilities for the hidden state and observables."""
    _hmm_table(
        start_prob=hmm.startprob_,
        trans_prob=hmm.transmat_,
        emission_prob=hmm.emissionprob_,
        state_names=states,
        observable_names=observables,
    )
