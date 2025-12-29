import sentencepiece as spm
from pathlib import Path
import os
from enum import Enum
import yaml

PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR   = os.path.join(PARENT_DIR, '01_data')
SENTENCE_PIECE_DATA = os.path.join(DATA_DIR, 'sentencepiece')
CONFIG_FILE = r'config.yaml'

class Language(Enum):
    ENG = 0
    KOR = 1

def build_tokenizer_model(language: Language, input_file):
    with open(CONFIG_FILE, "r") as f:

        cfg = yaml.safe_load(f)
    CHAR_COVERAGE = cfg["Tokenizer"][language.name.lower()]["coverage"]
    VOCAB_SIZE    = cfg["Tokenizer"][language.name.lower()]["vocab_size"]
    MODEL_TYPE    = cfg["Tokenizer"]["algorithm"]

    # Train the tokenizer
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix="{}_{}".format(language.name.lower(), MODEL_TYPE),
        vocab_size=VOCAB_SIZE,              
        model_type=MODEL_TYPE,
        character_coverage=CHAR_COVERAGE,  
        pad_id=3 # By default, unk_id, bos_id & eos_id are 0, 1 & 2 respectively    
    )

def load_tokenizer_model(path_to_model: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor(model_file=path_to_model)
    return sp

def get_lines(file_path: str):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def main():
    build_tokenizer_model(
        Language.KOR, 
        os.path.join(SENTENCE_PIECE_DATA, "{}.txt".format('korean')) # IMPORTANT: Replace this filename with yours
    )
    build_tokenizer_model( 
        Language.ENG, 
        os.path.join(SENTENCE_PIECE_DATA, "{}.txt".format('english'))
    )

    # ENG_MODEL_PATH = 'english_unigram.model'
    # KOR_MODEL_PATH = 'korean_unigram.model'
    # eng_sp = load_tokenizer_model(ENG_MODEL_PATH)
    # kor_sp = load_tokenizer_model(KOR_MODEL_PATH)

if __name__ == '__main__':
    main()