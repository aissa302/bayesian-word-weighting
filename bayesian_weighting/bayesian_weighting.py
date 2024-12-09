from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

def calculate_bayes_word_weights(descriptions, labels):
    """
    Calculates Bayes-based word weights for each class in a text classification task.

    Args:
        descriptions (list of str): List of text descriptions (documents).
        labels (list of str): List of labels corresponding to the descriptions.

    Returns:
        dict: A dictionary containing word weights for each class.
    """
    # Step 1: Preprocess and Tokenize
    def tokenize(text):
        return text.lower().split()

    # Step 2: Count document frequencies for each word per class
    class_word_doc_counts = defaultdict(Counter)  # To store counts of docs containing each word per class
    class_doc_counts = Counter()  # To store total document counts per class

    for desc, label in tqdm(zip(descriptions, labels), desc="Processing Documents"):
        cls = label
        words = set(tokenize(desc))  # Use a set to avoid double-counting words within the same document
        class_word_doc_counts[cls].update(words)
        class_doc_counts[cls] += 1

    # Step 3: Calculate P(class)
    total_docs = sum(class_doc_counts.values())
    class_probabilities = {cls: count / total_docs for cls, count in class_doc_counts.items()}

    # Step 4: Calculate P(word | class) and P(class | word)
    total_word_counts = Counter()
    for cls in class_word_doc_counts:
        total_word_counts.update(class_word_doc_counts[cls])

    # Calculate dictionaries for P(class | word)
    word_weights = {cls: {} for cls in class_doc_counts.keys()}

    for word in tqdm(total_word_counts, desc="Calculating Word Weights"):
        for cls in class_word_doc_counts:
            # P(word | class) = count(docs containing word in class) / total docs in class
            if class_doc_counts[cls] > 0:  # Avoid division by zero
                word_given_class = class_word_doc_counts[cls][word] / class_doc_counts[cls]
            else:
                word_given_class = 0

            # P(class | word) using Bayes' theorem
            p_word = total_word_counts[word] / total_docs
            if p_word > 0:
                class_given_word = (word_given_class * class_probabilities[cls]) / p_word
            else:
                class_given_word = 0

            # Clip probabilities and calculate log-odds
            p_class_given_word = np.clip(class_given_word, 1e-10, 1 - 1e-10)
            log_odds = np.log(p_class_given_word / (1 - p_class_given_word))

            # Store in the respective dictionary
            word_weights[cls][word] = log_odds

    return word_weights

