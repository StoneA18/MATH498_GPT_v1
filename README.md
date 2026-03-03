# README: 

This repository contains code and an interface for creating, training, and querying a naive generative pretrained transformer from scratch.
Built as part of MATH498 - Decoding GPT. Colorado School of Mines, Spring 2026.

**Authors:** Jesan Ahammed Ovi, Stone Amsbaugh  
**Instructor:** Michael Ivanitsky

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Modules](#modules)
  - [Config](#config)
  - [Vocab](#vocab)
  - [MLP](#mlp)
  - [Attention](#attention)
  - [Transformer](#transformer)
  - [LanguageModel](#languagemodel)
  - [GPT](#gpt)
- [Training a Model](#training-a-model)
- [Querying a Model](#querying-a-model)
- [Saving and Loading](#saving-and-loading)
- [Datasets](#datasets)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Collaboration](#collaboration)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

In MATH498, we have covered the basic mathematical foundation on top of which GPTs (Generative Pretrained Transformers) are constructed. In this assignment, we utilize PyTorch and the math we have learned to implement a GPT language model.

## Repository Contents

For the purposes of navigating this repository, consider the following contents:

* README.md: That is this document. It describes the project and documents its functionality. Take a look around.
* master_utility.py: This document defines all classes and functions that can be used to create, train and query a GPT. All math and logic behind the model lives here.
* gpt_demo.ipynb: This is a **very** useful demonstration on how to use the GPT interface to train and use your own GPT model.
* pyproject.toml, uv.lock: This file is created by uv, the package manager used to develop this project. Among other things, it contains the dependencies for this project.
* texts: This directory should contain text files used for training the model. It comes with 'recipes.txt', which is a text dataset of many recipes that can be used for training (the text has many non-standard characters, and as a result performance is poor). If you use the default functionality to get the state of the union addresses for training the model, they will be downloaded to this directory.
* models: This direcotry contains saved models and vocabulary files. The models are the actual parameters that the model learns, and the vocabulary file contains the tokens it has learned and their IDs. Both of these are saved/loaded together when calling save/load functions, so you don't need to worry about them separately.
* unit_tests.py: This file contains unit tests that test all of the functionality defined in master_utility.py.
* other: There is largely no need to examine this directory. This directory contains old python files, notebooks, and other work that was used to develop and experiment as this project was created. It is saved as a reference for the work process and in case we want to dig up an old feature.

## Using the Project

Here we will describe the high-level functionality of the project, which lives in 'master_utility.txt'. This is what you need to use our interface, without understanding the inner workings.

This section will be very similar to 'gpt_demo.ipynb', which is perhaps a better resource for learning how to use this interface.

### Environment Setup

We recommend running this project with uv, but it is not required. If you opt not to use uv, ensure that your environment is set up with the packages listed as dependencies in 'pyproject.toml'.

To install and get started with uv, see: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

You can run:
```
uv sync
```
To install the project dependencies.

Before running a notebook, like 'gpt_demo.ipynb', you will need to select the kernel to use (top-right corner in VSCode), which will be under the virtual environment created by uv. To run a normal python script that you may creates, use:
```
uv run my_script.py
```

To run the tests, use:
```
uv run pytest unit_tests.py
```

If you are going to experiment with the following usage instructions, it is recommended you do so in a Jupyter notebook.

### Initializing a Model

Your model starts with a GPT object, which can be initialized with:
```
my_gpt = GPT()
```
### Training a Model

You can train a model by providing the training method the number of iterations and a text file to train on. 
```
my_gpt.train(n_iter = 1000, text_file='texts/recipes.txt')
```
#### Notes:
* You can use the following built-in method to grab and downlaod a text file containing every State of the Union address through 2024. 
```
path = Vocab.get_state_of_union_file()
```
The 'path' symbol can then be passed to the training method as the 'text_file' argument.
* The training text must be at least as long as max_seq_len, which unless you modify is 256. For ideal results, the training text should be much larger than this.

#### Options:
These additional options are also implemented, and can be used if desired:
* plain_text: If plain_text (a string) is set and text_file is not, the model will train on your string
* update: Every 'update' number of iterations, the program will print an update including the iteration number and the loss. If set to 0, no updates will be given. Defaults to 50.
* plot: If True, will display a plot of the training loss over each iteration at the end of training. Default is True.
* sample_prompt: If set, will query the model before and after training with this prompt (a string) and print the results. Default is None, in which case no querying will be done during training.

### Querying a Model

You can query a model with the method:
```
my_gpt.query("YOUR QUERY HERE", out_tokens=15)
```
The return will be a string, the models response.

#### Notes:
* If the query contains tokens the model has never seen, they will be ignored.

#### Options:
* out_tokens: An integer, the number of tokens (loosely similar to the number of words) that will be outputted from the model. Default is 15.
* sampling: One of "multinomial" or "greedy". "greedy" sampling will always choose the next token that has the highest computed probability, and is therefore repetetive and prone to being caught in loops. "multinomial" sampling is the default, and chooses a random token weighted on their computed probabilities, resulting in much more natural results.
* to_print: A boolean, default is False, but if True it will print the result in addition to simply returning it.

### Saving a Model
A model can be saved with:
```
my_gpt.save('my_gpt')
```
This only takes one parameter, the name of the model. 

#### Notes:
* The model name is not a path, just a string 'name' for the model. The model parameters will be saved to './models/{name}.mdl' and the model's vocabular will be saved to './models/{name}.vcb'.
* The user does not need to worry about these saved files or the fact that they are separate, they will be loaded correctly with just the name.

### Loading a Model
To load a previously saved model, we initialize a GPT object with the static method:
```
my_gpt = GPT.load_gpt_from('my_gpt')
```
Again, this only takes one parameter, the model name as a string that you wish to load.

### Chatting with a Model
To 'chat' with a model (which is nothing more than a nice interface to repeatedly query a model and see results), use:
```
my_gpt.chat()
```
Minus the prompt (and to_print), it takes the same parameters as the query method.

When run as a python script, this will initiatie a command line interface where you can converse with your model. When run inside of a Jupyter notebook, it will produce a widget that emulates the same interface.

## Project Structure

### Config
The config class allows an object to be defined that contains the hyperparameters for the model.

### Vocab
### Transformers, MLP, Attention, LanguageModel classes
### GPT Object

## Evaluation

This section is for grading purposes, as this repository is also an assignment (Hi Michael). 

## Collaboration

...who did what, any tools or references used, acknowledgements...

## Attribution

This repository contains recipes data that comes from this [repository](https://github.com/josephrmartinez/recipe-dataset). It is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The project also has a builtin function to download State of the Union text from [this kaggle dataset](https://www.kaggle.com/datasets/nicholasheyerdahl/state-of-the-union-address-texts-1790-2024). It has the folllowing [MIT license](https://www.mit.edu/~amini/LICENSE.md).