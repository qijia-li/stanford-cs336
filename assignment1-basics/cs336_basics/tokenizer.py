"""
BPE (Byte Pair Encoding) Tokenizer: inference (Tokenizer) and training (BPETokenizer).

- Tokenizer: encode/decode using vocab and merges (from_files, encode, decode, encode_iterable).
- BPETokenizer: train byte-level BPE (BPETokenizer.train) with heap-based merge selection.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
from collections import Counter
from typing import Dict, Iterator, Iterable, List, Tuple

import regex as re

from cs336_basics.constants import DEFAULT_CHUNK_SIZE, PAT

# ----- Inference helpers and Tokenizer -----


def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"
    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]


def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes = list(word_bytes)

    while True:
        min_token_id = float("inf")
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        word_bytes = (
            word_bytes[:best_pair_idx]
            + [merged]
            + word_bytes[best_pair_idx + 2 :]
        )

    return tuple(word_bytes)


def word2bytes(word):
    """Convert word string to tuple of bytes."""
    a = list(word.encode("utf-8"))
    return tuple(bytes([i]) for i in a)


def encode_merged(text, merges, vocab_to_id):
    word_list = re.findall(PAT, text)
    tokens = []
    for word in word_list:
        word_bytes = word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens


class Tokenizer:
    """BPE tokenizer for inference: encode/decode with vocab and merges."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode("utf-8") for i in self.special_tokens]

        self.vocab_to_id = {v: k for k, v in vocab.items()}

        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)
            vocab = {
                int(k): bytes(v, "latin1") if isinstance(v, str) else bytes(v)
                for k, v in vocab_data.items()
            }

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            lines = mf.readlines()
            merge_pairs = [
                tuple(line.strip().split())
                for line in lines
                if not line.startswith("#") and line.strip()
            ]
            merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode("utf-8")])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily yield token IDs for an iterable of strings (e.g. file handle)."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        return b"".join([self.vocab[t] for t in ids]).decode("utf-8", errors="replace")


# ----- BPE training: logging and BPETokenizer -----

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, info: bool = False):
    """Configure logging for the BPE tokenizer."""
    level = logging.DEBUG if debug else (logging.INFO if info else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


class BPETokenizer:
    """BPE Tokenizer class for training byte-level BPE tokenizers (heap-based merge)."""

    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    @staticmethod
    def train(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        **kwargs,
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """Train a byte-level BPE tokenizer on the given input file."""
        info_mode = os.getenv("BPE_INFO", "false").lower() in ("true", "1", "yes")
        debug_mode = os.getenv("BPE_DEBUG", "false").lower() in ("true", "1", "yes")
        setup_logging(debug=debug_mode, info=info_mode)

        if info_mode:
            logger.info("========== BPE training started ==========")
            logger.info(
                "输入: %s, vocab_size=%s, special_tokens=%s",
                input_path,
                vocab_size,
                special_tokens,
            )

        vocab, curr_next_token, existed_tokens = BPETokenizer._initialize_vocab(
            vocab_size, special_tokens
        )
        raw_words = BPETokenizer._pretokenize_file(
            input_path, special_tokens=special_tokens
        )
        if not raw_words:
            logger.warning(f"No words found after pretokenization from {input_path}")
            return vocab, []

        if info_mode:
            logger.info(
                "Step 1 tokenization: %d tokens (only regex matches, no special)",
                len(raw_words),
            )
            logger.info("Top 20 tokens: %s", raw_words[:20])

        word_byte_freqs = BPETokenizer._convert_to_byte_representation(raw_words)
        if not word_byte_freqs:
            logger.warning("No word byte frequencies after conversion")
            return vocab, []

        if info_mode:
            logger.info(
                "Step 2 convert to byte representation: %d different byte sequences -> frequencies",
                len(word_byte_freqs),
            )

        merges = BPETokenizer._train_bpe_merges(
            vocab,
            word_byte_freqs,
            vocab_size,
            curr_next_token,
            existed_tokens,
            example_mode=info_mode,
        )

        if info_mode:
            logger.info(
                "========== BPE training completed: %d merges ==========", len(merges)
            )

        return vocab, merges

    @staticmethod
    def _initialize_vocab(
        vocab_size: int, special_tokens: List[str]
    ) -> Tuple[Dict[int, bytes], int, set]:
        vocab = {i: bytes([i]) for i in range(256)}
        curr_next_token = 256
        existed_tokens = set(vocab.values())
        for special_token in special_tokens:
            if len(vocab) >= vocab_size:
                break
            vocab[curr_next_token] = special_token.encode("utf-8")
            curr_next_token += 1
            existed_tokens.add(special_token.encode("utf-8"))
        return vocab, curr_next_token, existed_tokens

    @staticmethod
    def _chunk_documents_streaming(
        input_path: str,
        chunk_size: int,
        special_token: str,
    ):
        leftover = ""
        token_len = len(special_token)
        with open(input_path, "r", encoding="utf-8") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                block = leftover + block
                leftover = ""
                last_eot_idx = block.rfind(special_token)
                if last_eot_idx == -1:
                    leftover = block
                else:
                    yield block[: last_eot_idx + token_len]
                    leftover = block[last_eot_idx + token_len :]
        if leftover:
            yield leftover

    @staticmethod
    def _pretokenize_file(
        input_path: str, special_tokens: List[str] = None
    ) -> List[str]:
        raw_words = []
        special_tokens = special_tokens or []
        special_for_chunk = (
            special_tokens[0] if special_tokens else "\x00"
        )

        def get_chunks():
            if not special_tokens:
                with open(input_path, "r", encoding="utf-8") as f:
                    yield f.read()
                return
            for chunk in BPETokenizer._chunk_documents_streaming(
                input_path, DEFAULT_CHUNK_SIZE, special_for_chunk
            ):
                yield chunk

        try:
            if special_tokens:
                sorted_special = sorted(special_tokens, key=len, reverse=True)
                union = "|".join(re.escape(t) for t in sorted_special)
                special_set = set(special_tokens)
            else:
                union = None
                special_set = set()

            for chunk in get_chunks():
                if not chunk:
                    continue
                if not special_tokens:
                    raw_words.extend(re.findall(PAT, chunk))
                    continue
                parts = re.split(f"({union})", chunk)
                for part in parts:
                    if not part or part in special_set:
                        continue
                    raw_words.extend(re.findall(PAT, part))
        except Exception as e:
            logger.error(f"Error pretokenizing the input file: {e}", exc_info=True)
            return []

        return raw_words

    @staticmethod
    def _convert_to_byte_representation(
        raw_words: List[str],
    ) -> Dict[Tuple[bytes, ...], int]:
        word_freqs = Counter(raw_words)
        word_byte_freqs = {}
        for word, freq in word_freqs.items():
            word_bytes = word.encode("utf-8")
            word_bytes_tuple = tuple(bytes([b]) for b in word_bytes)
            word_byte_freqs[word_bytes_tuple] = freq
        return word_byte_freqs

    @staticmethod
    def _train_bpe_merges(
        vocab: Dict[int, bytes],
        word_byte_freqs: Dict[Tuple[bytes, ...], int],
        vocab_size: int,
        curr_next_token: int,
        existed_tokens: set,
        example_mode: bool = False,
    ) -> List[Tuple[bytes, bytes]]:
        merges = []
        skipped_pairs = set()

        if example_mode:
            logger.info(
                "Step 3 start BPE iterations: Merge the most frequent byte pair each time"
            )

        iteration = 0
        while len(vocab) < vocab_size:
            iteration += 1
            pair_counts = BPETokenizer._count_byte_pairs(word_byte_freqs)
            if not pair_counts:
                logger.debug("No more pairs to merge")
                break

            max_count = max(pair_counts.values()) if pair_counts else 0
            heap = [(-c, p) for p, c in pair_counts.items()]
            heapq.heapify(heap)
            found_valid_pair = False
            candidates = []
            while heap and heap[0][0] == -max_count:
                _, pair = heapq.heappop(heap)
                candidates.append(pair)
            candidates.sort(key=lambda p: (p[0], p[1]), reverse=True)

            if logger.isEnabledFor(logging.DEBUG) and 54 < iteration < 80:
                logger.debug(f"Iteration {iteration}: Top 10 pairs by frequency:")
                for idx, p in enumerate(candidates[:10]):
                    try:
                        p1_str = p[0].decode("utf-8", errors="replace")
                        p2_str = p[1].decode("utf-8", errors="replace")
                        logger.debug(
                            f"  {idx}: ({p1_str!r}, {p2_str!r}) = {max_count}"
                        )
                    except Exception:
                        logger.debug(f"  {idx}: {p} = {max_count}")

            for pair in candidates:
                if pair in skipped_pairs:
                    continue
                merged_token = pair[0] + pair[1]
                if merged_token in existed_tokens:
                    skipped_pairs.add(pair)
                    continue
                most_frequent_pair = pair
                found_valid_pair = True
                if example_mode:
                    t1 = most_frequent_pair[0].decode("utf-8", errors="replace")
                    t2 = most_frequent_pair[1].decode("utf-8", errors="replace")
                    merged_str = merged_token.decode("utf-8", errors="replace")
                    logger.info(
                        "Iteration %d: Merge (%r, %r) -> %r (frequency=%d)",
                        iteration,
                        t1,
                        t2,
                        merged_str,
                        max_count,
                    )
                break

            if not found_valid_pair:
                logger.debug("All remaining pairs create existing tokens")
                break

            if len(vocab) >= vocab_size:
                break

            vocab[curr_next_token] = merged_token
            curr_next_token += 1
            existed_tokens.add(merged_token)
            merges.append(most_frequent_pair)
            word_byte_freqs = BPETokenizer._apply_merge(
                word_byte_freqs, most_frequent_pair, merged_token
            )

        return merges

    @staticmethod
    def _count_byte_pairs(
        word_byte_freqs: Dict[Tuple[bytes, ...], int],
    ) -> Counter:
        pair_counts = Counter()
        for word_bytes_tuple, freq in word_byte_freqs.items():
            for i in range(len(word_bytes_tuple) - 1):
                pair = (word_bytes_tuple[i], word_bytes_tuple[i + 1])
                pair_counts[pair] += freq
        return pair_counts

    @staticmethod
    def _apply_merge(
        word_byte_freqs: Dict[Tuple[bytes, ...], int],
        pair_bytes: Tuple[bytes, bytes],
        merged_token: bytes,
    ) -> Dict[Tuple[bytes, ...], int]:
        new_word_byte_freqs = {}
        for word_bytes_tuple, freq in word_byte_freqs.items():
            new_word_bytes = []
            i = 0
            while i < len(word_bytes_tuple):
                if (
                    i < len(word_bytes_tuple) - 1
                    and word_bytes_tuple[i] == pair_bytes[0]
                    and word_bytes_tuple[i + 1] == pair_bytes[1]
                ):
                    new_word_bytes.append(merged_token)
                    i += 2
                else:
                    new_word_bytes.append(word_bytes_tuple[i])
                    i += 1
            new_word_bytes_tuple = tuple(new_word_bytes)
            if new_word_bytes_tuple in new_word_byte_freqs:
                new_word_byte_freqs[new_word_bytes_tuple] += freq
            else:
                new_word_byte_freqs[new_word_bytes_tuple] = freq
        return new_word_byte_freqs
