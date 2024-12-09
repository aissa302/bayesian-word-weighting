
# Bayesian Word Weighting for Text Classification

## Overview

This repository implements a novel Bayesian word weighting technique for text classification. Traditional TF-IDF approaches assign a single weight to a word across all categories, which may not reflect its category-specific importance. To address this limitation, we propose a Bayesian-based word weighting method that assigns unique weights to words based on their importance within each category, leveraging Bayes' theorem.

## Key Features

- **Bayesian Weighting**: Calculates word weights by considering category-specific distributions, ensuring more accurate representation for classification tasks.
- **Word2Vec Integration**: Combines domain-specific Word2Vec embeddings with Bayesian weights for a robust feature representation.
- **Enhanced Classification Performance**: Handles unique category words effectively, even when they appear in only a few documents.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/bayesian-word-weighting.git
   cd bayesian-word-weighting
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Dataset
Ensure your dataset is in a suitable format (e.g., CSV) with columns for text and corresponding category labels.

### 2. Train Word2Vec Model
Train a domain-specific Word2Vec model or use a pre-trained one:
```python
from gensim.models import Word2Vec

# Load or train your Word2Vec model
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
```

### 3. Calculate Bayesian Weights
Compute the prior, likelihood, and posterior probabilities for words and categories using the formulas provided in the [algorithm description](docs/algorithm.md).

Use the provided algorithm to create word weights embeddings for your text data:
```python
from bayesian_weighting import calculate_bayes_word_weights

weights_matrix = calculate_bayes_word_weights(documents, categories)
```
### 4. Generate Weighted Emebeddings
To use the generated bayes-based weights, you need to generate the embedding vectors of each word using custom or pretrained word embedding such as word2vec, glove and fastext, then multiply each word embedding vector by its corresponding bayesian weight to get the weighted word vectors.

embedding_matrix = $embedding_vector(w_i) * weights_matrix(w_i)$

### 5. Train Your Model
Pass the weighted embeddings to your classification model (e.g., BiLSTM):
```python
model.fit(embedding_matrix, labels)
```

## Algorithm

For a detailed explanation of the Bayesian word weighting algorithm, see the [Algorithm Documentation](docs/algorithm.md).

## Example

See the [example notebook](examples/bayesian_word_weighting_example.ipynb) for a step-by-step demonstration of how to use the algorithm.

## File Structure

```
bayesian-word-weighting/
├── examples/                       # Example usage notebooks
│   └── bayesian_word_weighting_example.ipynb
├── docs/                           # Documentation files
│   └── algorithm.md
├── bayesian_weighting
│   └── __init__.py
│   └── bayesian_weighting.py       # Main algorithm implementation
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
└── README.md                       # This README file
```

## Contributing

We welcome contributions! If you have ideas for improvements or find a bug, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please reach out to `ai.benyahya@edu.umi.ac.ma`.

