markdown
Copy code
# Named Entity Recognition (NER) with Neural Networks

This repository demonstrates the implementation of a Named Entity Recognition (NER) model using Neural Networks. The model is built with an LSTM-based architecture and is trained on a dataset to identify and classify named entities in text.

## Getting Started

### Prerequisites

To run this project, you need to have Python installed along with the following packages:

- pandas
- numpy
- scikit-learn
- tensorflow
- spacy
- matplotlib
- keras

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow spacy matplotlib keras
You will also need to download the SpaCy English language model:

bash
Copy code
python -m spacy download en_core_web_sm
Dataset
The dataset used for this task can be downloaded here. It contains sentences with words and their corresponding POS and Tag labels. Make sure to save it as ner_dataset.csv in your working directory.

Data Preparation
python
Copy code
from google.colab import files
uploaded = files.upload()
import pandas as pd
data = pd.read_csv('ner_dataset.csv', encoding='unicode_escape')
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
python
Copy code
data['Word_idx'] = data['Word'].map(token2idx)
data['Tag_idx'] = data['Tag'].map(tag2idx)
data_fillna = data.fillna(method='ffill', axis=0)
data_group = data_fillna.groupby(['Sentence #'], as_index=False)['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
python
Copy code
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def get_pad_train_test_val(data_group, data):
    n_token = len(list(set(data['Word'].to_list())))
    n_tag = len(list(set(data['Tag'].to_list())))

    tokens = data_group['Word_idx'].tolist()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value=n_token - 1)

    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value=tag2idx["O"])
    n_tags = len(tag2idx)
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]
    
    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.1, train_size=0.9, random_state=2020)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, tags_, test_size=0.25, train_size=0.75, random_state=2020)

    print(
        'train_tokens length:', len(train_tokens),
        '\ntrain_tokens length:', len(train_tokens),
        '\ntest_tokens length:', len(test_tokens),
        '\ntest_tags:', len(test_tags),
        '\nval_tokens:', len(val_tokens),
        '\nval_tags:', len(val_tags),
    )
    
    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags

train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(data_group, data)
Model Training
python
Copy code
import numpy as np
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

def get_bilstm_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(token2idx)+1, output_dim=64, input_length=max([len(s) for s in data_group['Word_idx'].tolist()])))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat'))
    model.add(LSTM(units=64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    model.add(TimeDistributed(Dense(len(tag2idx), activation="relu")))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model_bilstm_lstm = get_bilstm_lstm_model()
python
Copy code
def train_model(X, y, model):
    loss = list()
    for i in range(25):
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
        loss.append(hist.history['loss'][0])
    return loss

results = pd.DataFrame()
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
Model Testing
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
