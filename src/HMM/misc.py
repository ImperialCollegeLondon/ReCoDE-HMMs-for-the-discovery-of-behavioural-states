"""Module for miscellaneous functions used in the notebooks or plotting functions."""

from math import floor

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM
from tabulate import tabulate


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


def create_chain(states: list[str], length) -> list:
    """Creates a sequence according to Markovian principles.

    Args:
        states: A list of len==3, containing the name of the states to be generated
        length: The length of the generated sequence

    Returns:
        A python list of len==length of the given states given the transition rates set in the function

    """
    raise RuntimeError(
        f"The length of states sould be 3, not {len(states)}"
    )

    mat_trans = [[0.7, 0.05, 0.25], [0.35, 0.5, 0.15], [0.05, 0.3, 0.65]]

    i = 0

    markov_chain = [states[0]]
    mood = states[0]

    while i != length:
        for c, state in enumerate(states):
            if mood == state:
                mood = np.random.choice(states, replace=True, p=mat_trans[c])
                markov_chain.append(mood)
                break

        i += 1

    return markov_chain


def bin_data(data: pd.DataFrame, column: str, bin_column: str, function: str, bin_secs: int) -> pd.DataFrame:
    """A method to bin all data points to a larger time bin, then summarise a column.
    
    Args:
        data: the pandas dataframe with the time series data
        column: the column the will be aggregated
        bin_column: the column with the time series data
        function: the aggregating function to be used on the grouped time series
        bin_secs: the time (in seconds) that the new dataframe will be binned to

    Returns:
        A modified pandas data frame with two columns, the new time series and 
        the aggragated column named f"{column}_{function}"

    """
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


def _hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
    """Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object."""
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


def hmm_display(hmm: CategoricalHMM, states: list[str], observables: list[str]) -> None:
    """Prints the transion probabilities for the hidden state and observables.
    
    Args:
        hmm: the trained hmm whose matrices will be displayed
        states: a list of the given names of the states for the model
        observables: a list of the given names of the emissions for the model
    
    Returns:
        None

    """
    _hmm_table(
        start_prob=hmm.startprob_,
        trans_prob=hmm.transmat_,
        emission_prob=hmm.emissionprob_,
        state_names=states,
        observable_names=observables,
    )
