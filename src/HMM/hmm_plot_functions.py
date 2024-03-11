import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from functools import partial 
from math import floor, ceil
import random 

from misc import rle, bin_data

def hmm_mean_length(state_array, delta_t = 60, raw = False):
    """
    Finds the mean length of each state run per array/fly 
    returns a dataframe with a state column containing the states id and a mean_length column
    params:
    @state_array =  1D numpy array produced from a HMM decoder
    @delta_t = the time difference between each element of the array
    @raw = If true then length of all runs of each stae are returned, rather than the mean
    """
    assert(isinstance(raw, bool))
    delta_t_mins = delta_t / 60

    v, s, l = rle(state_array)

    df = pd.DataFrame(data = zip(v, l), columns = ['state', 'length'])
    df['length_adjusted'] = df['length'].map(lambda l: l * delta_t_mins)
    
    if raw == True:
        return df
    else:
        gb_bout = df.groupby('state').agg(**{
                            'mean_length' : ('length_adjusted', 'mean')
        })
        gb_bout.reset_index(inplace = True)

        return gb_bout

def hmm_pct_state(state_array, time, total_states, avg_window = 30):
    """
    Takes a window of n and finds what percentage each state is present within that window
    returns a dataframe with columns t, and states with their corresponding percentage per window
    params:
    @state_array =  1D numpy array produced from a HMM decoder
    @time = 1D numpy array of the timestamps of state_array of equal length and same order
    @total_states = numerical array denoting the states in 'state_array'
    @avg_window = length of window given as elements of the array
    """
    states_dict = {}

    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    for i in total_states:
        states_dict['state_{}'.format(i)] = moving_average(np.where(state_array == i, 1, 0), n = avg_window)

    adjusted_time = time[avg_window-1:]

    df = pd.DataFrame.from_dict(states_dict)
    df.insert(0, 't', adjusted_time)
                        
    return df

def hmm_decode(d, h, b, var, fun, t= 't', return_type = 'array'):

    # bin the data to 60 second intervals with a selected column and function on that column
    bin_df = d.groupby('id', group_keys = False).apply(partial(bin_data,
                                                                column = var, 
                                                                bin_column = t,
                                                                function = fun, 
                                                                bin_secs = b
    ))

    gb = bin_df.groupby(bin_df.index)[f'{var}_{fun}'].apply(list)
    time_list = bin_df.groupby(bin_df.index)['t_bin'].apply(list)

    states_list = []
    df = pd.DataFrame()

    for i, t, id in zip(gb, time_list, time_list.index):
        seq_o = np.array(i)
        seq = seq_o.reshape(-1, 1)
        logprob, states = h.decode(seq)

        if return_type == 'array':
            states_list.append(states)
        if return_type == 'table':
            label = [id] * len(t)
            all = zip(label, t, states, seq_o)
            all = pd.DataFrame(data = all)
            df = pd.concat([df, all], ignore_index = False)

    return states_list, time_list

def plot_hmm_overtime(data, hmm, variable, labels, colours, wrapped = False, tbin = 60, func = 'max', avg_window = 30):
    """
    Creates a plot of all states overlayed with y-axis shows the liklihood of being in a sleep state and the x-axis showing time in hours.
    The plot is generated through the plotly package

    Args:
        hmm (hmmlearn trained hmm): This should be a trained HMM Learn object with the correct hidden states and emission states for your dataset
        @variable = string, the column heading of the variable of interest. Default is "moving"
        @labels = list[string], the names of the different states present in the hidden markov model. If None the labels are assumed to be ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake']
        @colours = list[string], the name of the colours you wish to represent the different states, must be the same length as labels. If None the colours are a default for 4 states (blue and red)
        It accepts a specific colour or an array of numbers that are acceptable to plotly
        @wrapped = bool, if True the plot will be limited to a 24 hour day average
        @tbin = int, the time in seconds you want to bin the movement data to, default is 60 or 1 minute
        @func = string, when binning to the above what function should be applied to the grouped data. Default is "max" as is necessary for the "moving" variable
        @avg_window, int, the window in minutes you want the moving average to be applied to. Default is 30 mins
    """

    data = data.copy(deep = True).reset_index()

    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError('The number of labels do not match the number of states in the model')

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)

    data = pd.DataFrame()
    for l, t in zip(states_list, time_list):
        tdf = hmm_pct_state(l, t, list(range(len(labels))), avg_window = int((avg_window * 60)/tbin))
        data = pd.concat([data, tdf], ignore_index = True)

    if wrapped is True:
        data['t'] = data['t'].map(lambda t: t % (60*60*24))

    data['t'] = data['t'] / (60*60)
    t_min = int(12 * floor(data.t.min() / 12))
    t_max = int(12 * ceil(data.t.max() / 12))    
    t_range = [t_min, t_max]  

    plt.figure(figsize=(10,8))

    for c, (col, n) in enumerate(zip(colours, labels)):

        column = f'state_{c}'

        gb_df = data.groupby('t').agg(**{
                    'mean' : (column, 'mean'), 
                    'SD' : (column, 'std'),
                    'count' : (column, 'count')
                })

        gb_df['SE'] = (1.96*gb_df['SD']) / np.sqrt(gb_df['count'])
        gb_df['y_max'] = gb_df['mean'] + gb_df['SE']
        gb_df['y_min'] = gb_df['mean'] - gb_df['SE']
        gb_df = gb_df.reset_index()

        plt.plot(gb_df['t'], gb_df['mean'], color = col, label = n)
        plt.fill_between(gb_df['t'], gb_df['y_min'], gb_df['y_max'], color=col, alpha=0.25)

    plt.legend(loc="upper right")
    plt.ylabel('% time in state')
    plt.xlabel('time (hours)')
    plt.ylim((0,1))

    plt.show()

def plot_hmm_quantify(data, hmm, variable, labels, colours, tbin = 60, func = 'max'):
    """
    
    """

    data = data.copy(deep = True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError('The number of labels do not match the number of states in the model')
    label_dict  = {k : v for k, v in zip(list_states, labels)}

    rows = []
    for array in states_list:
        unique, counts = np.unique(array, return_counts=True)
        row = dict(zip(unique, counts))
        rows.append(row)
    counts_all =  pd.DataFrame(rows)
    counts_all['sum'] = counts_all.sum(axis=1)
    counts_all = counts_all.iloc[:, list_states[0]: list_states[-1]+1].div(counts_all['sum'], axis=0)
    counts_all.fillna(0, inplace = True)

    plt.figure(figsize=(10,8))


    d = pd.melt(counts_all)
    d['State'] = d['variable'].map(label_dict)
    sns.set_style("whitegrid")
    ax = sns.swarmplot(data = d, x="State", y="value", hue = 'State', palette=colours, legend = False, size=3)
    ax = sns.boxplot(data = d, x="State", y="value", hue = 'State', palette=colours, showcaps=False, showfliers=False, whiskerprops={'linewidth':0})

    ax.set(ylabel = '% of time in state')
    ax.set(ylim=(0, 1))


    means = d.groupby(['variable'])['value'].mean().round(3)
    vertical_offset = d['value'].mean() * 0.4

    for xtick in ax.get_xticks():
        ax.text(xtick,means[xtick] + vertical_offset,means[xtick], 
                horizontalalignment='center',size='x-large',color='black',weight='bold')

def plot_hmm_quantify_length(data, hmm, variable, labels, colours, tbin = 60, func = 'max'):
    
    data = data.copy(deep = True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(labels):
        raise RuntimeError('The number of labels do not match the number of states in the model')
    label_dict  = {k : v for k, v in zip(list_states, labels)}

    d = pd.DataFrame()
    for l in states_list:
        length = hmm_mean_length(l, delta_t = tbin) 
        d = pd.concat([d, length], ignore_index= True)

    d['State'] = d['state'].map(label_dict)


    plt.figure(figsize=(10,10))
    sns.set_style("whitegrid")

    ax = sns.swarmplot(data = d, x="State", y="mean_length", hue = 'State', palette=colours, legend = False, size=3)
    ax = sns.boxplot(data = d, x="State", y="mean_length", hue = 'State', palette=colours, showcaps=False, showfliers=False, whiskerprops={'linewidth':0})

    ax.set(ylabel = 'mean length of state')
    ax.set(yscale="log")


    means = round(d.groupby(['state'])['mean_length'].mean())
    vertical_offset = d['mean_length'].median() * 0.25

    for xtick in ax.get_xticks():
        ax.text(xtick,means[xtick] + vertical_offset,means[xtick], 
                horizontalalignment='center',size='x-large',color='black',weight='bold')

def plot_hmm_raw(data, hmm, variable, colours, tbin = 60, func = 'max'):
        
    data = data.copy(deep = True).reset_index()

    states_list, time_list = hmm_decode(data, hmm, tbin, variable, func)
    time_list = list(time_list)
    list_states = list(range(len(hmm.transmat_)))
    if len(list_states) != len(colours):
        raise RuntimeError('The number of colours do not match the number of states in the model')
    
    rand_ind = random.choice(list(range(0,len(states_list))))

    st = states_list[rand_ind]
    time = time_list[rand_ind]
    time = np.array(time) / 86400

    for c, i in enumerate(colours):
        if c == 0:
            col = np.where(st == c, colours[c], np.NaN)
        else:
            col = np.where(st == c, colours[c], col)

    plt.figure(figsize=(80,10))

    plt.scatter(time, st, s = 50*2, marker='o', c=col)
    plt.plot(time, st, marker='o', markersize = 0, mfc='white', mec='white', c='black', lw = 1, ls = '-')

    plt.xlabel('Time (days)')
    plt.ylabel('State')