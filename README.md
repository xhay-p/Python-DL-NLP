# Python, Data Science and Deep Learning for NLP
## Requirements
1. [Python3](https://www.python.org/)
2. [Ipython](https://ipython.org/index.html)
3. [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
4. [Numpy](https://pypi.org/project/numpy/)
5. [Matplotlib](https://pypi.org/project/matplotlib/)
6. [Seaborn](https://seaborn.pydata.org/)
7. [Pandas](https://pypi.org/project/pandas/)
9. [PyTorch](https://pytorch.org/)
10. [NLTK](https://www.nltk.org/)
11. [Spacy](https://spacy.io/)
12. [Huggingface Transformers](https://huggingface.co/docs/transformers/index)
13. [Sentence transformers](https://www.sbert.net/)
## Installation

**Linux (Ubuntu)**
1. Download and install Python3: `sudo apt-get install python3`
2. Download and install pip3: `sudo apt-get install python3-pip`
3. Download and install virtual environment: `pip3 install virtualenv`
4. Update: `sudo apt-get update`
5. Download the file **setup.sh** & **requirements.txt**
6. `chmod +x setup.sh`
7. `./setup.sh`
9. Well Done! You are ready for hands-on-sessions

**Mac OS**
1. Follow the instructions given in the link below, [For installing python 3 and virtual environment](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-macos)
2. Download the file **setup.sh** & **requirements.txt**
3. `chmod +x setup.sh`
4. `./setup.sh`
6. Well Done! You are ready for hands-on-sessions.

**Windows**
1. [Download Python3 latest version](https://www.python.org/downloads/windows/)
2. Most distribution comes with pip, [if not](https://www.liquidweb.com/kb/install-pip-windows/)
3. Download and Install Virtual Environment `pip install virtualenv`
4. [Creating and Activating Virtual Environment](https://programwithus.com/learn-to-code/Pip-and-virtualenv-on-Windows/)
5. Installing necessary libraries in virtual environment: `pip install -r requirements.txt ` <br>
`python -m spacy download en_core_web_sm` <br>
Check https://www.nltk.org/data.html for NLTK required data.
6. Well Done! You are ready for hands-on-sessions.

**NOTE** 
1. You can also download all the above-mentioned necessary tools and framework using [Anaconda](https://www.anaconda.com/distribution/).
2. DL libraries installation corresponds to CPU version. Check respective websites(link mentioned above) for GPU installation.

## Topics Covered
1. **Python Basics** - Data-types, Variables, Operators and Operations, Conditionals (if, if-else, if-elif-else), Loops (for and while), Exception Handling, Strings, Lists, Dictionaries, Tuples, File & I/O, command-line and os-module. <br>

*File:*  python_basics/basic-python.ipynb

2. **Object-Oriented Concepts** - Classes, Objects, Classes and Functions, Methods, Inheritence.

*File:*  python_basics/oops.ipynb

3. **Numpy** - primitive data-types support in numpy, ndarry, creating array, basic operations and in-built functions, reshaping, multi-dimension array, indexing, slicing, broadcasting, linear algebra.

*File:*  python_basics/numpy.ipynb

4. **Visualization (using matplotlib and seaborn)** - plotting a figure, adding labels to the axis, adding title, adding legends, adding grids, multiple plots using subplot, scatter plot, step plot, bar plot, histogram, reading .csv through pandas.

*File:*  python_basics/matplotlib-seaborn.ipynb

5. **Pandas** - Creation, indexing and slicing, manipulation, grouping, sorting and few usefull functionalities associated with DatFrames and Series.

*File:*  python_basics/pandas.ipynb

6. **PyTorch** - Covers pyTorch basic on creating different types of tensors in cpu and gpu, basic linear algebra operation, some statistical operation, arithmetic operations on tensors, Variable and autograd differention basics.

*Folder:* pytorch_basics

7. **Text Processing using NLTK and Spacy** - Covers basic text pre-processing and feature exploration using both nltk and spacy suite. [Work in Progress]

*File:*  text_feature_exploration.ipynb

8. **Word Embeddings using Word2Vec** - Covers loading and exploring pre-trained Word2Vec embeddings using Gensim. [Work in Progress]

*File:*  word_embeddings_gensim_word2vec.ipynb

9. **Transformers** - Covers using huggingface transformers library. It covers pipeline module, models, tokenizer and other concepts from hub for some major NLP tasks such as *Sentiment Analysis*, *Text Generation*, *Named Entity Recognition*, *Question Answering*, and *Summarization*.
	1. Tokenizer
	2. Model
	3. Pipeline
	4. Trainer
	5. Quantization

*Folder:*  transformers_basics/
