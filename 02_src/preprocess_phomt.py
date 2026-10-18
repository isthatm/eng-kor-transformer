import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import unicodedata
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, TextIO

import sentencepiece as spm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_INPUT_DIR = WORKSPACE_ROOT / "Ref" / "PhoMT" / "PhoMT" / "detokenization"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "01_data" / "PhoMT_cleaned"
DEFAULT_FILTERED_DIR = PROJECT_ROOT / "01_data" / "PhoMT_filtered"
DEFAULT_ENGLISH_TOKENIZER = PROJECT_ROOT / "01_data" / "sentencepiece" / "viet-sentencepiece" / "english_phomt.model"
DEFAULT_VIETNAMESE_TOKENIZER = PROJECT_ROOT / "01_data" / "sentencepiece" / "viet-sentencepiece" / "vietnamese_phomt.model"
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    """Parse the processing stage, corpus paths, tokenizer paths, and filtering options."""
    parser = argparse.ArgumentParser(description="Clean or length-filter the PhoMT corpus.")
    parser.add_argument("--stage", choices=("clean", "filter"), required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--english-tokenizer", type=Path, default=DEFAULT_ENGLISH_TOKENIZER)
    parser.add_argument("--vietnamese-tokenizer", type=Path, default=DEFAULT_VIETNAMESE_TOKENIZER)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_directories(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve stage-specific defaults; args contains the selected stage and optional path overrides."""
    if args.stage == "clean":
        input_dir = args.input_dir or DEFAULT_RAW_INPUT_DIR
        output_dir = args.output_dir or DEFAULT_CLEAN_DIR
    else:
        input_dir = args.input_dir or DEFAULT_CLEAN_DIR
        output_dir = args.output_dir or DEFAULT_FILTERED_DIR
    return input_dir, output_dir


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    """Load a SentencePiece model; model_path identifies the model and must define BOS/EOS tokens."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    if tokenizer.bos_id() < 0 or tokenizer.eos_id() < 0:
        raise ValueError(f"Tokenizer must define BOS and EOS tokens: {model_path}")
    return tokenizer


def normalize_text(text: str) -> str:
    """Trim and normalize text to Unicode NFC; text is one corpus sentence."""
    return unicodedata.normalize("NFC", text.strip())


def pair_digest(english: str, vietnamese: str) -> bytes:
    """Create a stable pair digest; english and vietnamese are normalized aligned sentences."""
    english_bytes = english.encode("utf-8")
    vietnamese_bytes = vietnamese.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(english_bytes).to_bytes(8, "big"))
    digest.update(english_bytes)
    digest.update(vietnamese_bytes)
    return digest.digest()


def validate_paths(input_dir: Path, output_dir: Path, overwrite: bool, stage: str) -> None:
    """Validate corpus inputs and output conflicts; arguments specify directories, overwrite behavior, and stage."""
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different")

    missing = []
    conflicts = []
    for split in SPLITS:
        for language in ("en", "vi"):
            source = input_dir / split / f"{split}.{language}"
            target = output_dir / split / f"{split}.{language}"
            if not source.is_file():
                missing.append(source)
            if target.exists() and not overwrite:
                conflicts.append(target)

    if stage == "clean" and not overwrite:
        for filename in ("cleaning_summary.json", "removed_pairs.jsonl"):
            target = output_dir / filename
            if target.exists():
                conflicts.append(target)

    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(map(str, missing)))
    if conflicts:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them:\n"
            + "\n".join(map(str, conflicts))
        )


def aligned_pairs(input_dir: Path, split: str) -> Iterator[tuple[int, str, str]]:
    """Yield aligned source lines; input_dir contains split files and split selects their names."""
    english_path = input_dir / split / f"{split}.en"
    vietnamese_path = input_dir / split / f"{split}.vi"
    sentinel = object()

    with (
        english_path.open("r", encoding="utf-8") as english_file,
        vietnamese_path.open("r", encoding="utf-8") as vietnamese_file,
    ):
        for line_number, pair in enumerate(
            zip_longest(english_file, vietnamese_file, fillvalue=sentinel),
            start=1,
        ):
            english, vietnamese = pair
            if english is sentinel or vietnamese is sentinel:
                raise ValueError(f"Alignment error in {split} at line {line_number}")
            yield line_number, english, vietnamese


def output_paths(temporary_dir: Path, split: str) -> tuple[Path, Path]:
    """Create and return temporary output paths; arguments identify the temporary root and split."""
    split_dir = temporary_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{split}.en", split_dir / f"{split}.vi"


def log_removed_pair(
    log_file: TextIO,
    split: str,
    line_number: int,
    reason: str,
    english: str,
    vietnamese: str,
) -> None:
    """Write one removed pair as JSONL; arguments identify its source, reason, and normalized text."""
    record = {
        "split": split,
        "line_number": line_number,
        "reason": reason,
        "english": english,
        "vietnamese": vietnamese,
    }
    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_clean_split(
    split: str,
    input_dir: Path,
    temporary_dir: Path,
    connection: sqlite3.Connection,
    removed_log: TextIO,
) -> dict[str, int]:
    """Clean and deduplicate one split; arguments provide paths, deduplication storage, and the removal log."""
    target_english, target_vietnamese = output_paths(temporary_dir, split)
    stats = {"read": 0, "empty": 0, "duplicate": 0, "written": 0}

    with (
        target_english.open("w", encoding="utf-8", newline="\n") as english_out,
        target_vietnamese.open("w", encoding="utf-8", newline="\n") as vietnamese_out,
    ):
        for line_number, raw_english, raw_vietnamese in aligned_pairs(input_dir, split):
            stats["read"] += 1
            if stats["read"] % 100_000 == 0:
                connection.commit()

            english = normalize_text(raw_english)
            vietnamese = normalize_text(raw_vietnamese)
            if not english or not vietnamese:
                stats["empty"] += 1
                log_removed_pair(
                    removed_log,
                    split,
                    line_number,
                    "empty_pair",
                    english,
                    vietnamese,
                )
                continue

            result = connection.execute(
                "INSERT OR IGNORE INTO seen_pairs (digest) VALUES (?)",
                (pair_digest(english, vietnamese),),
            )
            if result.rowcount == 0:
                stats["duplicate"] += 1
                log_removed_pair(
                    removed_log,
                    split,
                    line_number,
                    "duplicate_pair",
                    english,
                    vietnamese,
                )
                continue

            english_out.write(english + "\n")
            vietnamese_out.write(vietnamese + "\n")
            stats["written"] += 1

    connection.commit()
    return stats


def process_filter_split(
    split: str,
    input_dir: Path,
    temporary_dir: Path,
    english_tokenizer: spm.SentencePieceProcessor,
    vietnamese_tokenizer: spm.SentencePieceProcessor,
    max_tokens: int,
) -> dict[str, int]:
    """Length-filter one cleaned split; arguments provide paths, tokenizers, and the BOS/EOS-inclusive limit."""
    target_english, target_vietnamese = output_paths(temporary_dir, split)
    stats = {"read": 0, "too_long": 0, "written": 0}

    with (
        target_english.open("w", encoding="utf-8", newline="\n") as english_out,
        target_vietnamese.open("w", encoding="utf-8", newline="\n") as vietnamese_out,
    ):
        for line_number, raw_english, raw_vietnamese in aligned_pairs(input_dir, split):
            stats["read"] += 1
            english = raw_english.rstrip("\r\n")
            vietnamese = raw_vietnamese.rstrip("\r\n")
            if not english or not vietnamese:
                raise ValueError(f"Empty pair in cleaned {split} data at line {line_number}")

            english_length = len(english_tokenizer.encode(english, out_type=int)) + 2
            vietnamese_length = len(vietnamese_tokenizer.encode(vietnamese, out_type=int)) + 2
            if english_length > max_tokens or vietnamese_length > max_tokens:
                stats["too_long"] += 1
                continue

            english_out.write(english + "\n")
            vietnamese_out.write(vietnamese + "\n")
            stats["written"] += 1

    return stats


def publish_outputs(temporary_dir: Path, output_dir: Path) -> None:
    """Publish completed files atomically; arguments identify temporary and final output directories."""
    for split in SPLITS:
        target_dir = output_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for language in ("en", "vi"):
            source = temporary_dir / split / f"{split}.{language}"
            target = target_dir / f"{split}.{language}"
            os.replace(source, target)

    for filename in ("cleaning_summary.json", "removed_pairs.jsonl"):
        source = temporary_dir / filename
        if source.exists():
            os.replace(source, output_dir / filename)


def write_cleaning_summary(
    path: Path,
    input_dir: Path,
    stats: dict[str, dict[str, int]],
) -> None:
    """Write cleaning statistics as JSON; arguments specify the destination, source corpus, and counters."""
    totals = {
        name: sum(split_stats[name] for split_stats in stats.values())
        for name in ("read", "empty", "duplicate", "written")
    }
    summary = {
        "stage": "clean",
        "input_dir": str(input_dir.resolve()),
        "normalization": "NFC",
        "deduplication_scope": "global_across_train_dev_test",
        "splits": stats,
        "totals": totals,
    }
    with path.open("w", encoding="utf-8", newline="\n") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")


def run_clean_stage(input_dir: Path, output_dir: Path) -> dict[str, dict[str, int]]:
    """Run cleaning for all splits; input_dir is raw PhoMT and output_dir receives cleaned files."""
    with tempfile.TemporaryDirectory(prefix=".phomt_clean_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        database_path = temporary_dir / "seen_pairs.sqlite3"
        removed_log_path = temporary_dir / "removed_pairs.jsonl"

        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE seen_pairs (digest BLOB PRIMARY KEY) WITHOUT ROWID")
            with removed_log_path.open("w", encoding="utf-8", newline="\n") as removed_log:
                stats = {
                    split: process_clean_split(
                        split,
                        input_dir,
                        temporary_dir,
                        connection,
                        removed_log,
                    )
                    for split in SPLITS
                }
        finally:
            connection.close()

        write_cleaning_summary(temporary_dir / "cleaning_summary.json", input_dir, stats)
        publish_outputs(temporary_dir, output_dir)
    return stats


def run_filter_stage(
    input_dir: Path,
    output_dir: Path,
    english_tokenizer: spm.SentencePieceProcessor,
    vietnamese_tokenizer: spm.SentencePieceProcessor,
    max_tokens: int,
) -> dict[str, dict[str, int]]:
    """Run token filtering for all splits; arguments provide cleaned data, output, tokenizers, and length limit."""
    with tempfile.TemporaryDirectory(prefix=".phomt_filter_", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        stats = {
            split: process_filter_split(
                split,
                input_dir,
                temporary_dir,
                english_tokenizer,
                vietnamese_tokenizer,
                max_tokens,
            )
            for split in SPLITS
        }
        publish_outputs(temporary_dir, output_dir)
    return stats


def print_stats(stage: str, stats: dict[str, dict[str, int]]) -> None:
    """Print stage statistics; stage names the operation and stats contains per-split counters."""
    for split, counters in stats.items():
        values = ", ".join(f"{name}={value:,}" for name, value in counters.items())
        print(f"{stage}/{split}: {values}")


def main() -> None:
    """Run the selected PhoMT cleaning or token-filtering stage."""
    args = parse_args()
    input_dir, output_dir = resolve_directories(args)
    validate_paths(input_dir, output_dir, args.overwrite, args.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "clean":
        stats = run_clean_stage(input_dir, output_dir)
    else:
        if args.max_tokens < 3:
            raise ValueError("--max-tokens must leave room for BOS, content, and EOS")
        english_tokenizer = load_tokenizer(args.english_tokenizer)
        vietnamese_tokenizer = load_tokenizer(args.vietnamese_tokenizer)
        stats = run_filter_stage(
            input_dir,
            output_dir,
            english_tokenizer,
            vietnamese_tokenizer,
            args.max_tokens,
        )

    print_stats(args.stage, stats)


if __name__ == "__main__":
    main()
