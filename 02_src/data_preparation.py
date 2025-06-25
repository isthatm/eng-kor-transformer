import sys
import time
import pandas as pd
import os
from pathlib import Path
import pickle
import re
import argparse
import string
from collections import Counter

from spacy.lang.en import English
from konlpy.tag import Okt

PARENT_DIR = ""
DATA_DIR   = ""

"""
    Use soynlp to build a custom tokenizer from a corpus file.
    Use Konlpy -> built-in tokenizer for Korean text.
"""
LANG_TOKENIZERS = dict()

class Vocab:
    SPECIALS = ['<pad>', '<unk>', '<sos>', '<eos>']
    
    def __init__(self, tokens, min_freq=2):
        """
            Create a vocabulary from a list of tokens.
            Args:
                tokens (list[str]): The list of tokens to build the vocabulary from.
                min_freq (int): Minimum frequency for a token to be included in the vocabulary.
                specials (list[str]): List of special tokens to include in the vocabulary.
        """
        self.counter = Counter(tokens)
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Add specials first
        for token in self.SPECIALS:
            self.add_token(token)
        
        for token, freq in self.counter.items():
            if freq >= min_freq: # Only include tokens with frequency >= min_freq
                self.add_token(token)
    
    def add_token(self, token):
        """
            Add a token to the vocabulary.
            Args:
                token (str): The token to add.
        """
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def __len__(self):
        return len(self.token_to_idx)

    def __getitem__(self, token):
        """
            Get the index of a token.
            Args:
                token (str): The token to get the index for.
            Returns:
                int: The index of the token, or the index of '<unk>' if the token is not found.
        """
        return self.token_to_idx.get(token, self.token_to_idx['<unk>'])

    def decode(self, idxs):
        """
            Decode a list of indices back to tokens.
            Args:
                idxs (list[int]): The list of indices to decode.
            Returns:
                list[str]: A list of tokens corresponding to the indices.
        """
        return [self.idx_to_token.get(idx, '<unk>') for idx in idxs]
    
    def most_common(self, n=10, skip_specials=True):
        """
        Return the n most common tokens (excluding specials if desired).
        Args:
            n (int): Number of top tokens to return.
            skip_specials (bool): Whether to exclude special tokens.
        Returns:
            list of tuples: [(token, freq), ...]
        """
        items = self.counter.most_common()
        if skip_specials:
            items = [(token, freq) for token, freq in items if token not in self.SPECIALS]
        return items[:n]
    
def kor_tokenize_fn(text: str) -> list:
    """
        Use Konlpy's Okt tokenizer for Korean text.
        Args:
            text (str): The Korean text to tokenize.
        Returns:
            list: A list of tokens extracted from the text.
    """
    tokenizer = Okt()
    tokens = tokenizer.morphs(text)

    return tokens

def eng_tokenize_fn(text: str) -> list:
    """
        Use spaCy's English tokenizer for English text.
        Args:
            text (str): The English text to tokenize.
        Returns:
            list: A list of tokens extracted from the text.
    """
    nlp = English()
    tokenizer = nlp.tokenizer
    tokens = tokenizer(text)

    return [token.text for token in tokens]

def build_vocab(df: pd.DataFrame, lang: str) -> Vocab:
    tokenizer = LANG_TOKENIZERS.get(lang)
    tokenized_sentences = [tokenizer(sentence) for sentence in df[lang] if isinstance(sentence, str)] 
    # E.g. [ 
    #   ['I', am', 'travelling', 'the', 'world'], 
    #   ['I', 'will', 'have', 'a', 'trip', 'around', 'the', 'world'] 
    # ]
    all_tokens = [token 
                  for sentence in tokenized_sentences 
                  for token in sentence]
    return Vocab(all_tokens, min_freq=2)

def load_data(train_data_file: str):
    df = pd.read_csv(train_data_file, encoding='utf-8')
    df = df.dropna()  # Drop rows with NaN values
    df = df[
            df['korean'].apply(lambda x: isinstance(x, str)) & 
            df['english'].apply(lambda x: isinstance(x, str))
        ]  # Ensure 'korean' column has string values
    
    # Clean the text in both columns
    df['korean'] = df['korean'].apply(clean_text)
    df['english'] = df['english'].apply(clean_text)

    # Drop rows that became empty
    df = df[
        (df['korean'].str.strip() != '') &
        (df['english'].str.strip() != '')
    ]
    return df

def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character

    Returns:
        normalized sentence
    """
    text = text.translate(str.maketrans(
        '', # src char (NOT USED)
        '', # dest char (NOT USED)
        string.punctuation # char to be removed
        ))     
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with a single space
    return text.strip()

def save_vocab(vocab: Vocab, file_path: str):
    """
        Save the vocabulary to a file.
        Args:
            vocab (Vocab): The vocabulary to save.
            file_path (str): The path to the file where the vocabulary will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(file_path: str) -> Vocab:
    """
        Load the vocabulary from a file.
        Args:
            file_path (str): The path to the file where the vocabulary is saved.
        Returns:
            Vocab: The loaded vocabulary.
    """
    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


if __name__ == "__main__":
    PARENT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = os.path.join(PARENT_DIR, '01_data')
    LANG_TOKENIZERS = {
        'korean': kor_tokenize_fn,
        'english': eng_tokenize_fn,
    }

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Tokenization script")
    parser.add_argument('--src_lang', type=str, default='english', help='Source language for tokenization (kor or eng)')
    parser.add_argument('--target_lang', type=str, default='korean', help='Target language for tokenization (kor or eng)')
    config = parser.parse_args() # Parse command line arguments -> config

    train_data_file = os.path.join(DATA_DIR, 'train.csv') # Path to training data file
    # Load training data
    df = load_data(train_data_file)
    print("Loaded training data with shape:", df.shape)

    # Build the vocabulary for both languages 
    test_eng1 = "I am   travelling the world"
    test_eng2 = "I will have       a   trip  around the world"
    test_kor1 = "나는 세계를 여행하고 있다"
    test_kor2 = "너가 열렬히 대쉬하면 그녀가 받아줄지!"

    eng_vocab_pickle = os.path.join(DATA_DIR, 'eng_vocab.pickle')
    eng_vocab = None
    if not os.path.exists(eng_vocab_pickle):
        print("Building English vocabulary...")
        eng_vocab = build_vocab(df, 'english')
        save_vocab(eng_vocab, os.path.join(eng_vocab_pickle))
    else:
        print("Loading existing English vocabulary...")
        eng_vocab = load_vocab(eng_vocab_pickle)
    print("English vocabulary size:", len(eng_vocab))
    print("Most common English tokens:", eng_vocab.most_common(10))

    kor_vocab_pickle = os.path.join(DATA_DIR, 'kor_vocab.pickle')
    kor_vocab = None
    if not os.path.exists(kor_vocab_pickle):
        print("Building Korean vocabulary...")
        kor_vocab = build_vocab(df, 'korean')
        save_vocab(kor_vocab, os.path.join(DATA_DIR, 'kor_vocab.pickle'))
    else:
        print("Loading existing Korean vocabulary...")
        kor_vocab = load_vocab(kor_vocab_pickle)
    print("Korean vocabulary size:", len(kor_vocab))
    print("Most common Korean tokens:", kor_vocab.most_common(10))

    # eng_tokens1 = [eng_vocab[token.lower()] for token in eng_tokenize_fn(test_eng1)]
    # eng_tokens2 = [eng_vocab[token.lower()] for token in eng_tokenize_fn(test_eng2)]
    # print(eng_tokens1)
    # print(eng_tokens2)
    kor_tokens_1 = kor_tokenize_fn(test_kor2)
    kor_num_tokens_1 = [kor_vocab[token] for token in kor_tokens_1]
    print(kor_tokens_1)
    print(kor_num_tokens_1)