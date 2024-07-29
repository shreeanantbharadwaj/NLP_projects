
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

\`\`\`bash
pip install pandas numpy scikit-learn tensorflow spacy matplotlib keras
\`\`\`

You will also need to download the SpaCy English language model:

\`\`\`bash
python -m spacy download en_core_web_sm
\`\`\`

### Dataset

The dataset used for this task can be downloaded [here](#). It contains sentences with words and their corresponding POS and Tag labels. Make sure to save it as \`ner_dataset.csv\` in your working directory.

## Usage

1. Clone the repository:

\`\`\`bash
git clone https://github.com/yourusername/NER-NeuralNetworks.git
cd NER-NeuralNetworks
\`\`\`

2. Ensure all the prerequisites are installed.

3. Run the training script:

\`\`\`bash
python train_model.py
\`\`\`

4. To evaluate the model, run:

\`\`\`bash
python evaluate_model.py
\`\`\`

## Project Structure

- \`train_model.py\`: Script to train the NER model.
- \`evaluate_model.py\`: Script to evaluate the trained model.
- \`ner_dataset.csv\`: The dataset file containing sentences with their corresponding POS and Tag labels.
- \`models/\`: Directory to save the trained models.
- \`results/\`: Directory to save evaluation results and plots.

## Results

The model performance metrics and visualizations will be stored in the \`results/\` directory.

## Contributing

If you want to contribute to this project, please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the \`LICENSE\` file for details.
