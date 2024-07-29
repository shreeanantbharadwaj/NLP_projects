Named Entity Recognition (NER) with Neural Networks
This repository demonstrates the implementation of a Named Entity Recognition (NER) model using Neural Networks. The model is built with an LSTM-based architecture and is trained on a dataset to identify and classify named entities in text.

Getting Started
Prerequisites
To run this project, you need to have Python installed along with the following packages:

pandas
numpy
scikit-learn
tensorflow
spacy
matplotlib
keras
You can install the required packages using pip:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow spacy matplotlib keras
You will also need to download the SpaCy English language model:

bash
Copy code
python -m spacy download en_core_web_sm
Dataset
The dataset used for this task can be downloaded here. It contains sentences with words and their corresponding POS and Tag labels. Make sure to save it as ner_dataset.csv in your working directory.

Data Preparation
Load Data:

Load the dataset using pandas:

python
Copy code
from google.colab import files
uploaded = files.upload()
import pandas as pd
data = pd.read_csv('ner_dataset.csv', encoding='unicode_escape')
Create Mappings:

Generate mappings for tokens and tags:

python
Copy code
from itertools import chain

def get_dict_map(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}
    
    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))
    
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

token2idx, idx2token = get_dict_map(data, 'token')
tag2idx, idx2tag = get_dict_map(data, 'tag')
Transform Data:

Convert tokens and tags into indices and group them by sentence:

python
Copy code
data['Word_idx'] = data['Word'].map(token2idx)
data['Tag_idx'] = data['Tag'].map(tag2idx)
data_fillna = data.fillna(method='ffill', axis=0)
data_group = data_fillna.groupby(['Sentence #'], as_index=False)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
Padding and Splitting Data:

Prepare the data for training by padding sequences and splitting into train, validation, and test sets:

python
Copy code
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def get_pad_train_test_val(data_group, data):
    # Implementation here
    ...

train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)
Model Training
Define Model:

Create the BiLSTM model architecture:

python
Copy code
import numpy as np
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

def get_bilstm_lstm_model():
    # Implementation here
    ...

model_bilstm_lstm = get_bilstm_lstm_model()
Train Model:

Train the model on the dataset:

python
Copy code
def train_model(X, y, model):
    # Implementation here
    ...

results = pd.DataFrame()
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
Model Testing
Test the trained model using SpaCy to visualize entity recognition:

python
Copy code
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
text = nlp('Hi, My name is Aman Kharwal \n I am from India \n I want to work with Google \n Steve Jobs is My Inspiration')
displacy.render(text, style='ent', jupyter=True)
Results
After training for 25 epochs, the model provides results on the NER task. You can evaluate the performance based on accuracy and loss metrics provided during training.

Contributing
Feel free to submit issues, contribute improvements, or request features. Pull requests are welcome.

License
This project is licensed under the MIT License.
