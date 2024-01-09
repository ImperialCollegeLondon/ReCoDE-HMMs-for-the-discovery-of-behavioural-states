from itertools import cycle
import numpy as np 
from math import floor, ceil
import pandas as pd

def rle(x):
    """
    Find runs of consecutive items in an array

    params:
    @x = ID numpy array  
    
    returns three arrays containg the run values, the start indices of the runs, and the lengths of the runs 
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
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

def bin_data(data, column, bin_column, function, bin_secs):
    """ a method that will bin all data poits to a larger time bin and then summarise a column """
    index_name = data['id'].iloc[0]

    data[bin_column] = data[bin_column].map(lambda t: bin_secs * floor(t / bin_secs))
    output_parse_name = f'{column}_{function}' # create new column name

    bout_gb = data.groupby(bin_column).agg(**{
        output_parse_name : (column, function)    
    })

    bin_parse_name = f'{bin_column}_bin'
    bout_gb.rename_axis(bin_parse_name, inplace = True)
    bout_gb.reset_index(level=0, inplace=True)
    old_index = pd.Index([index_name] * len(bout_gb.index), name = 'id')
    bout_gb.set_index(old_index, inplace =True)

    return bout_gb