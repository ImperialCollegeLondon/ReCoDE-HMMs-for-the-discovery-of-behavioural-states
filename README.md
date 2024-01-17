<!-- Your Project title, make it sound catchy! -->

# Hidden Markov Models for the discovery of behavioural states

<!-- Provide a short description to your project -->

## Description

This is an exemplar project to help you understand the concepts behind the Hidden Markov Model (HMM), how to implement 
one with the pythoin package hmmlearn, and finally how to explore the decoded data. 

HMM's are widely used in multiple fields, including biology, natural language processing, and finance as a predictor
of future states in a sequence. However, here we will be utilising the hidden model states to create a hypothesied
internal behavioural architecture.

This is all a part of the ReCoDE Project at Imperial College London

<!-- What should the students going through your exemplar learn -->

## Learning Outcomes

Only a basic understanding of python is needed prior to beginning, with the tuorials walking you through the 
use of numpy and pandas to curate data for use with the hmmlearn package.

- 1. Understanding the core concepts of HMMs
- 2. Curating data and training/validating your own HMM
- 3. Visualing and understanding your decoded data

| Task       | Time    |
| ---------- | ------- |
| Reading    | TBD hours |
| Practising | TBD hours |

## Requirements

### Academic

A basic knowledge of python is needed.

The tutorial will be based in numpy and pandas, two data science pacakages for working with and manipulating data.
However, previou knowledge will not be neccesary with all steps explained and options to explore parts further

No prior knowledge of HMMs is needed or deep understanding of mathmatical modelling. However, if you do want to read more about 
HMMs I found these resources very useful when starting out:
Hidden Markov Models - Speech and Language Processing  -> (https://web.stanford.edu/~jurafsky/slp3/A.pdf)


### System

| Program                  | Version                  |
| ------------------------ | ------------------------ |
| Python                   | >= 3.11.0                |


### Dependencies

| Package                  | Version                  |
| ------------------------ | ------------------------ |
| numpy                    | >= 1.26.3                |
| pandas                   | >= 2.1.4                 |
| hmmlearn                 | >= 0.3.0                 |
| jupyter                  | >= XXXXX                 |


## Getting Started

### Workflow

The tutorial will be taught through sequential jupyter notebooks which you will clone to your local computer.
The code will be mainly written out and executed from within the notebooks so you will get a feel for the full workflow to generate
and test HMMs. A few parts of the code that help the code run or tidy up the plots will be imported from elsewhere in the project (here),
there is also pre formed python class to train your HMM for use after the tutorial.

### Workstation

You'll need something to run and edit the code in the notebooks as we go along, this tutorial was created in [Visual Studio Code] (https://code.visualstudio.com/),
but you can use whatever code editor you like.

### Cloning the repository

If you don't have it already install git to your machine, see [here] (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for details on all OS's 

Once installed run the following comman in the terminal after moving to the location you want it saved

```bash
git clone https://github.com/ImperialCollegeLondon/ReCoDE-HMMs-for-the-discovery-of-behavioural-states.git HMM_tutorial
cd HMM_tutorial
```

## Project Structure

```log
.
├── examples
│   ├── ex1
│   └── ex2
├── src
|   ├── file1.py
|   ├── file2.cpp
|   ├── ...
│   └── data
├── app
├── docs
├── main
└── test
```

<!-- Change this to your License. Make sure you have added the file on GitHub -->

## License

This project is licensed under the [BSD-3-Clause license](LICENSE.md)
