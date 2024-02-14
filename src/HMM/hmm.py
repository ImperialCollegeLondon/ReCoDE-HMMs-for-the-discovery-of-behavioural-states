"""A module for training and testing hidden markov models on categorical data.

The module is built on top of the hmmlearn package.
"""
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmm_functions import hmm_pct_state
from hmmlearn import hmm
from misc import bin_data
from tabulate import tabulate  # type: ignore


class HMM(pd.DataFrame):
    """A class for training and testing hidden markov models on categorical data."""

    def __init__(self, data, index=None, columns=None, dtype=None, copy=True):
        """Initialise the HMM class."""
        super(HMM, self).__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )

    @staticmethod
    def _hmm_decode(d, h, b, var, fun, t="t", return_type="array"):
        # bin the data to 60 second intervals
        # with a selected column and function on that column.
        bin_df = d.groupby("id", group_keys=False).apply(
            partial(bin_data, column=var, bin_column=t, function=fun, bin_secs=b)
        )

        gb = bin_df.groupby(bin_df.index)[f"{var}_{fun}"].apply(list)
        time_list = bin_df.groupby(bin_df.index)["t_bin"].apply(list)

        # logprob_list = []
        states_list = []
        df = pd.DataFrame()

        for i, t, id in zip(gb, time_list, time_list.index):
            seq_o = np.array(i)
            seq = seq_o.reshape(-1, 1)
            logprob, states = h.decode(seq)

            # logprob_list.append(logprob)
            if return_type == "array":
                states_list.append(states)
            if return_type == "table":
                label = [id] * len(t)
                # previous_state = np.array(states[:-1], dtype = float)
                # previous_state = np.insert(previous_state, 0, np.nan)
                all = zip(label, t, states, seq_o)
                all = pd.DataFrame(data=all)
                df = pd.concat([df, all], ignore_index=False)

        if return_type == "array":
            return states_list, time_list  # , logprob_list
        if return_type == "table":
            df.columns = ["id", "bin", "state", var]
            return df

    @staticmethod
    def _hmm_table(
        start_prob, trans_prob, emission_prob, state_names, observable_names
    ):
        """Prints a formatted table.

        Prints a formatted table of the probabilities from a hmmlearn
        MultinomialHMM object.
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

    def hmm_display(self, hmm, states, observables):
        """Prints to screen the transion probabilities.

        Prints to screen the transion probabilities for the hidden state and
        observables for a given hmmlearn hmm object
        """
        self._hmm_table(
            start_prob=hmm.startprob_,
            trans_prob=hmm.transmat_,
            emission_prob=hmm.emissionprob_,
            state_names=states,
            observable_names=observables,
        )

    def hmm_train(
        self,
        states,
        observables,
        var_column,
        file_name,
        trans_probs=None,
        emiss_probs=None,
        start_probs=None,
        iterations=10,
        hmm_iterations=100,
        tol=50,
        t_column="t",
        bin_time=60,
        test_size=10,
        verbose=False,
    ):
        """A method to train a categorical HMM.

        A method to train a categorical HMM given a single column of data from multiple
        specimens. The data must be supplied in a integer categorical format, i.e. begin
        with your first observable represented as 0, your second as 1, and
        so on.

        The method is best used to train and subsequently test multiple HMM's that are
        intialised with differing transition probabilities, therefore provided
        probability matrices as a numpy array for the transission, emission, and
        starting matrices. Where the transition isn't 0 or 1 please replace it with
        'rand' which indicates to the function to replace it with a random nume, see
        example below for a 4 hidden state model.

        t_prob = np.array([['rand', 'rand', 'rand', 0.0],
                            ['rand', 'rand', 'rand', 0.0],
                            [0.0, 'rand', 'rand', 'rand'],
                            [0.0, 0.0, 'rand', 'rand']])

        There must be no NaNs in the training data.

        Resultant hidden markov models will be saved as a .pkl file
        The final trained model probability matrices will be printed to terminal at the
        end of the run time

        Args:
            states: (list of sting(s)). The names of hidden states for the model to
            train to.
            observables: (list of string(s)). The names of the observable states for the
            model to train to, it should be the same length as number of categories in
            training data.
            var_column: (string). The name of the column containing the data to train
            the model, should be augmented to be categorical integers.
            file_name: (string). The name of the .pkl file the resultant trained model
            will be saved to.
            trans_probs: (numpy array). The transtion probability matrix with shape
            'len(states) x len(states)', 0's restrict the model from training any
            tranisitons between those states. Default is None.
            emiss_probs: (numpy array). The emission probability matrix with shape
            'len(observables) x len(observables)', 0's same as above. Default is None.
            start_probs: (numpy array). The starting probability matrix with shape
            'len(states) x 0', 0's same as above. Default is None.
            iterations: (int). The number of times a new model will be trained from
            scratch. If you have 'rand' in your probability matrix these will be
            randomised at the start of each loop. Default is 10.
            hmm_iterations: (int). An argument to be passed to hmmlearn, the number of
            iterations to go through (in which the hmm transitions are updated) if the
            tol target isn't met. Default is 100
            tol: (int). The convergence threshold, EM will stop if the gain in
            log-likelihood is below this value. Default is 50
            t_column: (string). The name for the column containing the time series data,
            default is 't'
            bin_time: (int). The time in seconds the data time steps will be binned to
            before the training begins. Default is 60 (i.e 1 min)
            test_size: (int). The percentage size of the test dataset which the trained
            HMM will be score against. Default is 10.
            verbose: (bool, optional). An argument for hmmlearn, whether per-iteration
            convergence reports are printed to terminal. Default is False.

        Returns:
            A trained hmmlearn categoricalHMM object
        """
        if file_name.endswith(".pkl") is False:
            raise TypeError(
                "enter a file name and type (.pkl) for the hmm object to be saved under"
            )

        n_states = len(states)
        n_obs = len(observables)

        hmm_df = self.copy(deep=True)
        hmm_df.reset_index(inplace=True)

        gb = hmm_df.groupby("id")[var_column].apply(np.array)

        # split runs into test and train lists
        test_train_split = round(len(gb) * (test_size / 100))
        rand_runs = np.random.permutation(gb)
        train = rand_runs[test_train_split:]
        test = rand_runs[:test_train_split]

        len_seq_train = [len(ar) for ar in train]
        len_seq_test = [len(ar) for ar in test]

        seq_train = np.concatenate(train, 0)
        seq_train = seq_train.reshape(-1, 1)
        seq_test = np.concatenate(test, 0)
        seq_test = seq_test.reshape(-1, 1)

        for i in range(iterations):
            print(f"Iteration {i+1} of {iterations}")

            init_params = ""
            h = hmm.CategoricalHMM(
                n_components=n_states,
                n_iter=hmm_iterations,
                tol=tol,
                params="ste",
                verbose=verbose,
            )

            if start_probs is None:
                init_params += "s"
            else:
                s_prob = np.array(
                    [
                        [np.random.random() if y == "rand" else y for y in x]
                        for x in start_probs
                    ],
                    dtype=np.float64,
                )
                s_prob = np.array(
                    [[y / sum(x) for y in x] for x in s_prob], dtype=np.float64
                )
                h.startprob_ = s_prob

            if trans_probs is None:
                init_params += "t"
            else:
                # replace 'rand' with a new random number being 0-1
                t_prob = np.array(
                    [
                        [np.random.random() if y == "rand" else y for y in x]
                        for x in trans_probs
                    ],
                    dtype=np.float64,
                )
                t_prob = np.array(
                    [[y / sum(x) for y in x] for x in t_prob], dtype=np.float64
                )
                h.transmat_ = t_prob

            if emiss_probs is None:
                init_params += "e"
            else:
                # replace 'rand' with a new random number being 0-1
                em_prob = np.array(
                    [
                        [np.random.random() if y == "rand" else y for y in x]
                        for x in emiss_probs
                    ],
                    dtype=np.float64,
                )
                em_prob = np.array(
                    [[y / sum(x) for y in x] for x in em_prob], dtype=np.float64
                )
                h.emissionprob_ = em_prob

            h.init_params = init_params
            h.n_features = n_obs  # number of emission states

            # call the fit function on the dataset input
            h.fit(seq_train, len_seq_train)

            # Boolean output of if the number of runs convererged on set of appropriate
            # probabilites for s, t, an e
            print(
                "True Convergence:"
                + str(h.monitor_.history[-1] - h.monitor_.history[-2] < h.monitor_.tol)
            )
            print("Final log liklihood score:" + str(h.score(seq_train, len_seq_train)))

            if i == 0:
                try:
                    h_old = pickle.load(open(file_name, "rb"))
                    if h.score(seq_test, len_seq_test) > h_old.score(
                        seq_test, len_seq_test
                    ):
                        print("New Matrix:")
                        df_t = pd.DataFrame(h.transmat_, index=states, columns=states)
                        print(tabulate(df_t, headers="keys", tablefmt="github") + "\n")

                        with open(file_name, "wb") as file:
                            pickle.dump(h, file)
                except OSError:
                    with open(file_name, "wb") as file:
                        pickle.dump(h, file)

            else:
                h_old = pickle.load(open(file_name, "rb"))
                if h.score(seq_test, len_seq_test) > h_old.score(
                    seq_test, len_seq_test
                ):
                    print("New Matrix:")
                    df_t = pd.DataFrame(h.transmat_, index=states, columns=states)
                    print(tabulate(df_t, headers="keys", tablefmt="github") + "\n")
                    with open(file_name, "wb") as file:
                        pickle.dump(h, file)

            if i + 1 == iterations:
                h = pickle.load(open(file_name, "rb"))
                # print tables of trained emission probabilties, not accessible as
                # objects for the user
                self._hmm_table(
                    start_prob=h.startprob_,
                    trans_prob=h.transmat_,
                    emission_prob=h.emissionprob_,
                    state_names=states,
                    observable_names=observables,
                )
                return h

    def plot_hmm_overtime(
        self,
        hmm,
        variable,
        labels,
        colours,
        wrapped=False,
        tbin=60,
        func="max",
        avg_window=30,
    ):
        """Creates a plot of all states over time.

        Creates a plot of all states overlayed with y-axis shows the liklihood of being
        in a sleep state and the x-axis showing time in hours.The plot is generated
        through the plotly package.

        Args:
            hmm (hmmlearn trained hmm): This should be a trained HMM Learn object with
            the correct hidden states and emission states for your dataset.
            variable: string, the column heading of the variable of interest. Default is
            "moving".
            labels: list[string], the names of the different states present in the
            hidden markov model. If None the labels are assumed to be
            ['Deep sleep', 'Light sleep', 'Quiet awake', 'Full awake'].
            colours: list[string], the name of the colours you wish to represent the
            different states, must be the same length as labels. If None the colours are
            a default for 4 states (blue and red). It accepts a specific colour or an
            array of numbers that are acceptable to plotly.
            wrapped: bool, if True the plot will be limited to a 24 hour day average.
            tbin: int, the time in seconds you want to bin the movement data to, default
            is 60 or 1 minute.
            func: string, when binning to the above what function should be applied to
            the grouped data. Default is "max" as is necessary for the "moving"
            variable.
            avg_window: int, the window in minutes you want the moving average to be
            applied to. Default is 30 mins.
        """
        df = self.copy(deep=True).reset_index()

        states_list, time_list = self._hmm_decode(df, hmm, tbin, variable, func)

        df = pd.DataFrame()
        for state, t in zip(states_list, time_list):
            tdf = hmm_pct_state(
                state,
                t,
                list(range(len(labels))),
                avg_window=int((avg_window * 60) / tbin),
            )
            df = pd.concat([df, tdf], ignore_index=True)

        if wrapped is True:
            df["t"] = df["t"].map(lambda t: t % (60 * 60 * 24))

        df["t"] = df["t"] / (60 * 60)

        for c, (col, n) in enumerate(zip(colours, labels)):
            column = f"state_{c}"

            gb_df = df.groupby("t").agg(
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

            plt.plot(gb_df["t"], gb_df["mean"], color=col)

        return plt


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/lab/Desktop/ReCoDE-HMMs-for-the-discovery-of-behavioural-states/data/"
        "training_data_small.zip",
        index_col="id",
    )
    h = pickle.load(
        open(
            "/home/lab/Desktop/ReCoDE-HMMs-for-the-discovery-of-behavioural-states/"
            "data/test_hmm.pkl",
            "rb",
        )
    )
    hdf = HMM(data=df)
    hdf["moving"] = np.where(hdf["moving"] is True, 1, 0)
    hdf.hmm_train(
        states=["deep sleep", "light sleep", "quiet awake", "active awake"],
        observables=["immobile", "mobile"],
        var_column="moving",
        file_name="test_h.pkl",
        trans_probs=None,
        emiss_probs=None,
        start_probs=None,
        iterations=2,
        hmm_iterations=10,
        tol=50,
        t_column="t",
        bin_time=60,
        test_size=10,
        verbose=True,
    )

if __name__ == "__main__":
    df = pd.read_csv(
        "/home/lab/Desktop/ReCoDE-HMMs-for-the-discovery-of-behavioural-states/data/"
        "training_data_small.zip",
        index_col="id",
    )
    h = pickle.load(
        open(
            "/home/lab/Desktop/ReCoDE-HMMs-for-the-discovery-of-behavioural-states/"
            "data/test_hmm.pkl",
            "rb",
        )
    )
    hdf = HMM(data=df)
    hdf["moving"] = np.where(hdf["moving"] is True, 1, 0)
    hdf.plot_hmm_overtime(
        h,
        variable="moving",
        labels=["deep sleep", "light sleep", "quiet awake", "active awake"],
        colours=["darkblue", "lightblue", "orange", "red"],
        wrapped=True,
    )
