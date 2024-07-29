Copy code
# Named Entity Recognition (NER) with LSTM

This project demonstrates how to build and train a Named Entity Recognition (NER) model using LSTM networks. The model is trained on a dataset of text annotated with named entities, and the trained model is used to recognize entities in new text. This repository includes data preprocessing, model training, and testing steps.

## Project Overview

The NER model is built using a BiLSTM (Bidirectional Long Short-Term Memory) network with an additional LSTM layer. The project is implemented in Python using libraries such as `pandas`, `numpy`, `tensorflow`, and `spacy`.

## Dataset

The dataset used in this project can be downloaded [here](#). The dataset is assumed to be in CSV format with columns representing words and their associated tags (named entities).

## Requirements

Make sure you have the following Python packages installed:

- `pandas`
- `numpy`
- `tensorflow`
- `keras`
- `spacy`
- `google.colab`

You can install the required packages using pip:

```bash
pip install pandas numpy tensorflow keras spacy google.colab
Additionally, download the English language model for spaCy:

bash
Copy code
python -m spacy download en_core_web_sm
Data Preparation
Load the Data: The dataset is loaded into a pandas DataFrame and prepared for the neural network.

Mapping: Tokens and tags are converted into numerical indices using mapping dictionaries.

Padding and Splitting: The sequences are padded to ensure uniform length, and the data is split into training, validation, and test sets.

Model Architecture
The NER model is built using the following architecture:

Embedding Layer: Converts word indices into dense vectors.
Bidirectional LSTM Layer: Captures context from both directions.
LSTM Layer: Further processes the sequences.
TimeDistributed Dense Layer: Outputs predictions for each token.
Training
The model is trained for 25 epochs. The training function is set up to fit the model and track the loss.

Usage
Load and Prepare Data:

python
Copy code
from google.colab import files
import pandas as pd
uploaded = files.upload()
data = pd.read_csv('ner_dataset.csv', encoding='unicode_escape')
Data Preparation:

python
Copy code
# Code for preparing data...
Build and Train Model:

python
Copy code
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional, TimeDistributed
from keras.utils import plot_model

model = get_bilstm_lstm_model()
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model)
Testing:

python
Copy code
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
text = nlp('Hi, My name is Aman Kharwal \n I am from India \n I want to work with Google \n Steve Jobs is My Inspiration')
displacy.render(text, style='ent', jupyter=True)
Results
After training for 25 epochs, the model is evaluated for its performance on the test set and used to recognize entities in sample texts.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The dataset used in this project is sourced from [insert source here].
The architecture is inspired by common practices in NER tasks with LSTM networks.
