# Hidden Markov Models for the discovery of behavioural states

## Description

This is an exemplar project to help you understand the concepts behind the Hidden Markov Model (HMM), how to implement
one with the python package hmmlearn, and finally how to explore the decoded data.

HMMs are widely used in multiple fields, including biology, natural language processing, and finance as a predictor
of future states in a sequence. However, here we will be utilising the hidden model states to create a hypothesied
internal behavioural architecture.

The tutorial will also run briefly through how to clean and augment a real world dataset using numpy and pandas,
so that it's ready for training with hmmlearn.

The information in tutorial was primarily designed around the user completing reading along and completing a jupyter notebook in python.
But can followed loosely from just these pages. If reading along ignore any sections asking to complete any code (or complete it in your mind).

This is all a part of the ReCoDE Project at Imperial College London

## Learning Outcomes

Only a basic understanding of python is needed prior to beginning, with the tutorials walking you through the
use of numpy and pandas to curate data for use with the hmmlearn package.

- 1. Understanding the core concepts of HMMs
- 2. Curating data and training/validating your own HMM
- 3. Visualising and understanding your decoded data

## Requirements

### Academic

A basic knowledge of python is needed.

The tutorial will be based in numpy and pandas, two data science packages for working with and manipulating data.

No prior knowledge of HMMs is needed, nor deep understanding of mathmatical modelling. However, if you do want to read more about
HMMs, I found this resource very useful when starting out:
Hidden Markov Models - [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/A.pdf)

### System

| Program                  | Version                  |
| ------------------------ | ------------------------ |
| Python                   | >= 3.11.0                |
| Git                      | >= 2.43.0                |

### Packages

| Package                  | Version                  |
| ------------------------ | ------------------------ |
| numpy                    | >= 1.26.4                |
| pandas                   | >= 2.2.0                 |
| hmmlearn                 | >= 0.3.0                 |
| Matplotlib               | >= 3.8.3                 |
| seaborn                  | >= 0.13.2                |
| tabulate                 | >= 0.9.0                 |
| jupyter                  | >= 1.0.0                 |

## Getting Started

### Workflow

The tutorial will be taught through sequential jupyter notebooks which you will clone to your local computer.
The code will be mainly written out and executed from within the notebooks so you will get a feel for the full workflow to generate
and test HMMs. A few parts of the code that help the code run or tidy up the plots will be imported from elsewhere in the project.

In the folder src there is a jupyter notebook called notebook_answers.ipynb. This notebook contains the answers to parts of the notebook where you need to write your own code.

Once you've cloned the repo and installed the dependencies, open the first notebook as highlighted by the arrow in the structure below.

## Project Structure

```log
.
|
├── data
|   ├── example_hmm.pkl
|   ├── training_data_metadata.csv
|   └── training_data.zip
├── docs
├── notebooks
|   ├── 1_Understanding_HMMs.ipynb <---
|   ├── 2a_Cleaning_your_data.ipynb
|   ├── 2b_Training.ipynb
|   ├── 2c_Validating.ipynb
|   └── 3_Visualising_the_results.ipynb
└── src
    ├── hmm_plot_functions.py
    ├── misc.py
    └── notebook_answers.ipynb
```

### Workstation

You'll need something to run and edit the code in the notebooks as we go along. This tutorial was created in [Visual Studio Code](https://code.visualstudio.com/),
but you can use whatever code editor you like.

### Cloning the repository

If you don't have it already, install git to your machine, see [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for details on all OS's.

Once installed, run the following command in the terminal after moving to the location where you want it saved. We'll then 'cd' into the created folder.

```bash
git clone https://github.com/ImperialCollegeLondon/ReCoDE-HMMs-for-the-discovery-of-behavioural-states.git HMM_tutorial
cd HMM_tutorial
```

### Setting up your environment

```bash
python -m venv .venv
source .venv/bin/activate # with Powershell on Windows: `.venv\Scripts\Activate.ps1`
```

### Install requirements

```bash
pip install -r requirements.txt
```

You're now ready to jump into the first notebook. Open up the notebook 1_Understanding_HMMs.ipynb.

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
