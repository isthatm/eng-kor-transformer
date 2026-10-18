import argparse
from pathlib import Path

import sentencepiece as spm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "config-vie.yaml"
LANGUAGES = ("eng", "vie")


def parse_args() -> argparse.Namespace:
    """Parse the tokenizer configuration path and output overwrite option."""
    parser = argparse.ArgumentParser(description="Train English and Vietnamese PhoMT tokenizers.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load and validate tokenizer settings; config_path identifies the YAML configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("Tokenizer"), dict):
        raise ValueError(f"Missing Tokenizer configuration: {config_path}")
    return config["Tokenizer"]


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configured path; configured_path may be absolute or relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_tokenizer_model(language: str, config: dict, overwrite: bool) -> Path:
    """Train one SentencePiece tokenizer; arguments specify its language, settings, and overwrite policy."""
    if language not in config or not isinstance(config[language], dict):
        raise ValueError(f"Missing tokenizer settings for language: {language}")

    language_config = config[language]
    input_file = resolve_project_path(language_config["input_file"])
    output_dir = resolve_project_path(config["output_dir"])
    model_prefix = output_dir / language_config["model_prefix"]
    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")

    if not input_file.is_file():
        raise FileNotFoundError(f"Training data not found: {input_file}")
    if not overwrite and (model_path.exists() or vocab_path.exists()):
        raise FileExistsError(
            f"Tokenizer output already exists for {language}; pass --overwrite to replace it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=int(language_config["vocab_size"]),
        model_type=str(config["algorithm"]),
        character_coverage=float(language_config["coverage"]),
        byte_fallback=bool(config.get("byte_fallback", True)),
        hard_vocab_limit=bool(config.get("hard_vocab_limit", False)),
        input_sentence_size=int(config.get("input_sentence_size", 0)),
        shuffle_input_sentence=bool(config.get("shuffle_input_sentence", True)),
        pad_id=int(config.get("pad_id", 3)),
    )
    return model_path


def load_tokenizer_model(model_path: Path) -> spm.SentencePieceProcessor:
    """Load a SentencePiece tokenizer; model_path identifies the compiled model file."""
    return spm.SentencePieceProcessor(model_file=str(model_path))


def main() -> None:
    """Train the English and Vietnamese PhoMT tokenizers from the selected configuration."""
    args = parse_args()
    config = load_config(args.config)

    for language in LANGUAGES:
        model_path = build_tokenizer_model(language, config, args.overwrite)
        print(f"Created {language} tokenizer: {model_path}")


if __name__ == "__main__":
    main()
