import numpy as np
from typing import List
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence: str) -> List[str]:
    return nltk.word_tokenize(sentence)


def bag_of_words(tokenized_sentence: List[str], words: List[str]) -> np.ndarray:
    """
    возвращает вектор наличия слов из предложения в общем словаре
    """
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
