# Title Generation using Text Summarization: A Deep Learning Approach

## Table of contents
1. [About](#about)
2. [Data](#data)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Links](#links)

---

## About <a name="about"></a>
>Generate a headline/title prediction from descriptive text. The title will be a concise summary of the original input text. The pipeline consists of upstream natural language processing steps, and subsequently, a downstream Seq2Seq w/attention bidirectional neural network model for prediction. Parameters for both the NLP and model are configurable.

---

## Data <a name="data"></a>

> Input data for running the NLP should take on the form of a list of dictionaries. What you're trying to summarize should be keyed as "content" and the target as "title."

    [{"content":"input_1", "title":"target_1"} , ... , {"content":"input_1023", "title":"target_1023"}]

>
> Output processed data is the indexed feature vectors that are fed into the model. This data format is used for training and prediction.

    [{"content":[567,124,1210,55, 92], "title":[452, 95, 789] , ... , {"content":[5423,1234,156, 929], "title":[11,99,34]}]

## Installation <a name="installation"></a>
>General Requirements:
>
>* Clone the repo: git clone git@github.com:plotkiny/Deep-Learning.git
>* Download Anaconda for your operating system

>Create and activate Anaconda environment:

    conda create --name py35 python=3.5
    source activate py35

>Install Packages:
   
    conda install "package"
* jupyter  

    `conda install nb_conda` to link jupyter with Anaconda
* tensorflow and tensorflow-gpu, version=1.1.0
* scikit-learn
* tqdm
* pyenchant (install with pip)
* nltk

     `import nltk`  
     `nltk.download("tagsets")`  
     `nltk.download("punkt")`  
     `nltk.download("stopwords")`  

> Other Downloads:
* Download the semantic word embedding vectors: https://github.com/commonsense/conceptnet-numberbatch
>
>Optional Installation

* __screen__ for persistent terminal sessions
* __vim__ command-line text editor

---

## Usage <a name="usage"></a>
>The command-line arguments should be run from the src (root) directory


A.	Run the data through the processing pipelne.
```bash
python ./main/run.py -pre -con main/resources/configuration.txt -out /path_to_output_directory
```
B.	Train the model using the processed_data. The output directory should point to the directory where the processed_data file resides. 
```bash
python ./main/run.py -t -con main/resources/configuration.txt -out /path_to_output_directory
```
C.	Predict the model using the same processed_data. The output directory should point to the directory where the processed_data file resides. The train/test splits and random state can be changed by modifying the configuration file. 
```bash
python ./main/run.py -p -con main/resources/configuration.txt -out /path_to_output_directory
```
---

## Links <a name="links"></a>

`To learn more about LSTM networks:` http://colah.github.io/posts/2015-08-Understanding-LSTMs/

