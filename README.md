# Machine Learning Engineer Nanodegree - Capstone Project

**Date: February 2019 / Author: Bastian Carvajal Yañez**

This is my capstone project for the Udacity MLND. You can find the [proposal.pdf](proposal/proposal.pdf) and the [report.pdf](proposal/report.pdf) in this repository.


## Setup

This project was build with **Python 3.5**, on Ubuntu 16.04. Its recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create an environment for the project.

```
conda create -n capstone python=3.5
conda activate capstone
```

Then install the required python packages with PIP:

```
pip install -r requirements.txt
```

I've used an NVIDIA GTX 1060 GPU to run training faster. In this case use these file instead:

```
pip install -r requirements-gpu.txt
```


## Usage

#### Preprocess dataset

**Note:** This repo already contains the dataset and preprocessed files (including the GloVe vectors).

First you need to download the dataset from: [https://www.kaggle.com/nltkdata/conll-corpora/downloads/conll2002.zip/1](https://www.kaggle.com/nltkdata/conll-corpora/downloads/conll2002.zip/1), and unzip it in the `dataset/` folder. Then you can build the vocabulary and the Glove vector for the each dataset, with the next command:

```
cd dataset/
make
```

This will produce the following files:

```
├── conll2002
│   ├── esp
│   │   ├── glove.npz
│   │   ├── testa.tags.txt
│   │   ├── testa.words.txt
│   │   ├── testb.tags.txt
│   │   ├── testb.words.txt
│   │   ├── train.tags.txt
│   │   ├── train.words.txt
│   │   ├── vocab.chars.txt
│   │   ├── vocab.tags.txt
│   │   └── vocab.words.txt
│   ├── ned
│   │   ├── glove.npz
│   │   ├── testa.tags.txt
│   │   ├── testa.words.txt
│   │   ├── testb.tags.txt
│   │   ├── testb.words.txt
│   │   ├── train.tags.txt
│   │   ├── train.words.txt
│   │   ├── vocab.chars.txt
│   │   ├── vocab.tags.txt
│   │   └── vocab.words.txt
```

#### Train the model with active learning

**Note:** I provide pre-trained models, see the next section! ;)

To train the model use the next command:

```
python train.py -data_dir ./dataset/conll2002/esp -out_dir ./results/esp -sampling lc -run_counts 10
```

This will run 10 train steps, using _least confidence_ as the sampling method. This means that the model will be incrementally updated using more data. For each step the model will choose what annotated examples add to its train dataset.

At regular intervals the model will save checkpoint and train statistics, that you can visualize in _Tensorboard_. For each checkpoint a validation step will be executed on the corresponding `testa` dataset.

```
tensorboard --logdir ./results/esp
```

#### Test the model

If you do not want to train the models again, you should download the pre-trained models from: [saved_model_201902261744.tgz](https://drive.google.com/open?id=19xHOumQYKF-YDpNkcRD7p3-CyNQG0Jc6), and extract the `saved_model` folder.

Then, you can run the `predict.py` command passing it a text to process.

```
python predict.py saved_model/esp --line "Jose trabaja para CNN en Sao Paulo , Brasil ."
```

Then you can see something like this, were the model made its predictions:

```
words : Jose  trabaja para CNN   en Sao   Paulo , Brasil .
preds : B-LOC I-LOC   O    B-LOC O  B-LOC I-LOC O B-LOC  O
```

You can also try it with the testing dataset, the model will take a random example and compare its prediction with the true labels.

```
python predict.py saved_model/ned --sample_words dataset/conll2002/ned/testb.words.txt --sample_tags dataset/conll2002/ned/testb.tags.txt
```



