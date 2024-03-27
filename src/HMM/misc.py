"""Module for functions to analyse HMM outputs."""
from typing import Callable

from math import floor, ceil
import numpy as np
import pandas as pd
from pathlib import PurePath
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

def create_chain(states, start, length):

    if any([name == start for name in states]) is False:
        raise TypeError('The starting state name is not in the states list')

    mat_trans =   [[0.7, 0.05, 0.25],
                    [0.35, 0.5, 0.15],
                    [0.05, 0.3, 0.65]]

    i = 0

    markov_chain = [start]
    mood = start

    while i != length:

        for c, state in enumerate(states):

            if mood == state:
                mood = np.random.choice(states,replace=True,p=mat_trans[c])
                markov_chain.append(mood)
                break

        i += 1
    
    return markov_chain

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

def _hmm_table(start_prob, trans_prob, emission_prob, state_names, observable_names):
    """ 
    Prints a formatted table of the probabilities from a hmmlearn MultinomialHMM object
    """
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

def hmm_display(hmm: Type[hmmlearn.CategoricalHMM], states: list[str], observables: list[str]) -> None:
    """Prints the transion probabilities for the hidden state and observables."""
    _hmm_table(
        start_prob=hmm.startprob_,
        trans_prob=hmm.transmat_,
        emission_prob=hmm.emissionprob_,
        state_names=states,
        observable_names=observables,
    )
