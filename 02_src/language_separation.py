import csv
from pathlib import Path
import os
import re

'''
    This script reads the dataset from a .csv file of 2 columns (Language1 Language2)
    -> Create 2 .txt files, whose lines are sentences collected from Language1 & Language2
'''

def clean_text(text):
    """
    remove special characters from the input sentence to normalize it
    Args:
        text: (string) text string which may contain special character

    Returns:
        normalized sentence
    """
    # Keep: letters, digits, spaces, Korean Hangul, and punctuation (.,!?;:'"-)
    text = re.sub(r'[^\w\s\uAC00-\uD7A3.,!?;:\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text.strip()

PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PARENT_DIR, '01_data')
SENTENCE_PIECE_DATA = os.path.join(DATA_DIR, 'sentencepiece')

# INPUT .csv file path
input_file = os.path.join(DATA_DIR, 'test.csv')

# OUTPUT text file paths
korean_file  = os.path.join(SENTENCE_PIECE_DATA, 'test_korean.txt')
english_file = os.path.join(SENTENCE_PIECE_DATA, 'test_english.txt')

# Open all files
with open(input_file, newline='', encoding='utf-8') as csvfile, \
     open(korean_file, 'w', encoding='utf-8') as kor_out, \
     open(english_file, 'w', encoding='utf-8') as eng_out:

    reader = csv.reader(csvfile)
    
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue # skip the header row
        if len(row) < 2:
            continue  # skip malformed rows

        korean_sentence  = clean_text(row[0].strip())
        english_sentence = clean_text(row[1].strip())
        
        # Write to separate files (one sentence per line)
        kor_out.write(korean_sentence + '\n')
        eng_out.write(english_sentence + '\n')

print("DONE")
