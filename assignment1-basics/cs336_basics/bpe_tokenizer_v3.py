"""
BPE (Byte Pair Encoding) Tokenizer Training Implementation.

This module provides functionality for training a byte-level BPE tokenizer
on text corpora.
"""

import heapq
import logging
import multiprocessing
import os
import regex as re  # Use regex module for Unicode property escapes (\p{L}, \p{N})
from collections import Counter
from typing import Dict, List, Optional, Tuple

from cs336_basics.constants import DEFAULT_CHUNK_SIZE, PAT

# Multiprocessing: only use parallel when data is large enough to outweigh overhead.
# Set BPE_NUM_WORKERS > 1 (e.g. 4) to enable parallel processing for large corpora.
_PARALLEL_THRESHOLD_PRETOKENIZE = 2  # min number of chunks to use parallel pretokenize
_PARALLEL_THRESHOLD_WORD_FREQS = 2000  # min word_byte_freqs size for parallel count/apply
_DEFAULT_N_WORKERS = 1  # default 1 = sequential; set BPE_NUM_WORKERS to enable parallel

def _pretokenize_chunk(args: Tuple[str, Optional[str], set]) -> List[str]:
    """Worker: run regex pretokenization on one chunk. Used by multiprocessing."""
    chunk, union, special_set = args
    if union is None:
        return re.findall(PAT, chunk)
    parts = re.split(f"({union})", chunk)
    words = []
    for part in parts:
        if not part or part in special_set:
            continue
        words.extend(re.findall(PAT, part))
    return words


def _count_pairs_chunk(items_chunk: List[Tuple[Tuple[bytes, ...], int]]) -> Counter:
    """Worker: count byte pairs in a chunk of word_byte_freqs items."""
    pair_counts = Counter()
    for word_bytes_tuple, freq in items_chunk:
        for i in range(len(word_bytes_tuple) - 1):
            pair = (word_bytes_tuple[i], word_bytes_tuple[i + 1])
            pair_counts[pair] += freq
    return pair_counts


def _apply_merge_chunk(
    args: Tuple[List[Tuple[Tuple[bytes, ...], int]], Tuple[bytes, bytes], bytes]
) -> Dict[Tuple[bytes, ...], int]:
    """Worker: apply one merge to a chunk of word_byte_freqs items."""
    items_chunk, pair_bytes, merged_token = args
    new_word_byte_freqs = {}
    for word_bytes_tuple, freq in items_chunk:
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

# Create logger for this module
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, info: bool = False):
    """Configure logging for the BPE tokenizer.

    Args:
        debug: If True, set log level to DEBUG. Otherwise, set to WARNING.
        example: If True, set log level to INFO for readable example logs.
    """
    if info:
        level = logging.INFO
    else:
        level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

'''
This is the third multiprocessing version of the BPE tokenizer.
It optimizes the training process by
- using multiprocessing to parallelize the training process.
- using a min-heap to find the most frequent pair of bytes.
- using a dictionary to store the word byte frequencies.
- using a set to store the existing tokens.
- using a list to store the merges.
- using a function to count the byte pairs.
- using a function to apply the merge.
'''
class BPETokenizer:
    """BPE Tokenizer class for training byte-level BPE tokenizers."""

    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        """Initialize BPETokenizer.

        Args:
            input_path: Path to input text file for training
            vocab_size: Maximum vocabulary size
            special_tokens: List of special tokens to add to vocabulary
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens


    @staticmethod
    def train(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
        **kwargs
    ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        """Train a byte-level BPE tokenizer on the given input file.

        Args:
            input_path: Path to a text file with BPE tokenizer training data
            vocab_size: A positive integer that defines the maximum final vocabulary size
                (including the initial byte vocabulary, vocabulary items produced from merging,
                and any special tokens)
            special_tokens: A list of strings to add to the vocabulary. These special tokens
                do not otherwise affect BPE training
            **kwargs: Additional keyword arguments (unused).
            Parallel processing: set env BPE_NUM_WORKERS > 1 (e.g. 4) for large corpora.

        Returns:
            A tuple containing:
                - vocab: dict[int, bytes] - The tokenizer vocabulary, a mapping from int
                  (token ID in the vocabulary) to bytes (token bytes)
                - merges: list[tuple[bytes, bytes]] - A list of BPE merges produced from training.
                  Each list item is a tuple of bytes (<token1>, <token2>), representing that
                  <token1> was merged with <token2>. The merges are ordered by order of creation.
        """
        # Setup logging: BPE_INFO=1 for readable logs, BPE_DEBUG=1 for debug
        info_mode = os.getenv("BPE_INFO", "false").lower() in ("true", "1", "yes")
        debug_mode = os.getenv("BPE_DEBUG", "false").lower() in ("true", "1", "yes")
        setup_logging(debug=debug_mode, info=info_mode)

        if info_mode:
            logger.info("========== BPE training started ==========")
            logger.info("输入: %s, vocab_size=%s, special_tokens=%s", input_path, vocab_size, special_tokens)

        # Initialize vocabulary with byte vocabulary and special tokens
        vocab, curr_next_token, existed_tokens = BPETokenizer._initialize_vocab(
            vocab_size, special_tokens
        )
        # Pretokenize input file (pass special_tokens for intelligent boundary detection)
        raw_words = BPETokenizer._pretokenize_file(input_path, special_tokens=special_tokens)
        if not raw_words:
            logger.warning(f"No words found after pretokenization from {input_path}")
            return vocab, []

        if info_mode:
            logger.info("Step 1 tokenization: %d tokens (only regex matches, no special)", len(raw_words))
            logger.info("Top 20 tokens: %s", raw_words[:20])

        # Convert words to byte-level representation
        word_byte_freqs = BPETokenizer._convert_to_byte_representation(raw_words)
        if not word_byte_freqs:
            logger.warning("No word byte frequencies after conversion")
            return vocab, []

        if info_mode:
            logger.info("Step 2 convert to byte representation: %d different byte sequences -> frequencies", len(word_byte_freqs))

        # Perform BPE training iterations
        merges = BPETokenizer._train_bpe_merges(
            vocab, word_byte_freqs, vocab_size, curr_next_token, existed_tokens,
            example_mode=info_mode,
        )

        if info_mode:
            logger.info("========== BPE training completed: %d merges ==========", len(merges))

        return vocab, merges

    @staticmethod
    def _initialize_vocab(
        vocab_size: int, special_tokens: List[str]
    ) -> Tuple[Dict[int, bytes], int, set]:
        """Initialize vocabulary with byte vocabulary and special tokens.

        Args:
            vocab_size: Maximum vocabulary size
            special_tokens: List of special tokens to add

        Returns:
            Tuple of (vocab dict, next_token_id, set of existing tokens)
        """
        # Initialize with byte vocabulary (0-255)
        vocab = {i: bytes([i]) for i in range(256)}
        curr_next_token = 256
        existed_tokens = set(vocab.values())

        # Add special tokens
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
        """Yield chunks that end on a special-token boundary.

        When the file contains no special token, accumulates and yields the whole
        file at EOF. This ensures identical word counts to the reference.
        """
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
    def _pretokenize_file(input_path: str, special_tokens: List[str] = None) -> List[str]:
        """Pretokenize input file using regex pattern.

        Chunks end on last occurrence of special token (or whole file at EOF when none). 
        Only regex matches are added; special tokens are boundaries only, not in raw_words.
        Uses multiprocessing when there are multiple chunks.
        """
        special_tokens = special_tokens or []
        special_for_chunk = special_tokens[0] if special_tokens else "\x00"

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

            chunks_list = [c for c in get_chunks() if c]
            n_workers = int(os.getenv("BPE_NUM_WORKERS", _DEFAULT_N_WORKERS))

            if len(chunks_list) >= _PARALLEL_THRESHOLD_PRETOKENIZE and n_workers > 1:
                with multiprocessing.Pool(processes=n_workers) as pool:
                    chunk_args = [(c, union, special_set) for c in chunks_list]
                    chunk_results = pool.map(_pretokenize_chunk, chunk_args)
                raw_words = []
                for words in chunk_results:
                    raw_words.extend(words)
            else:
                raw_words = []
                for chunk in chunks_list:
                    raw_words.extend(_pretokenize_chunk((chunk, union, special_set)))
        except Exception as e:
            logger.error(f"Error pretokenizing the input file: {e}", exc_info=True)
            return []

        return raw_words

    @staticmethod
    def _convert_to_byte_representation(
        raw_words: List[str]
    ) -> Dict[Tuple[bytes, ...], int]:
        """Convert words to byte-level representation with frequencies.

        Args:
            raw_words: List of pretokenized words

        Returns:
            Dictionary mapping tuple of bytes to word frequency
        """
        word_freqs = Counter(raw_words)

        # Convert each word to a tuple of single-byte bytes objects
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
        """Perform BPE training by iteratively merging most frequent pairs.

        Args:
            vocab: Vocabulary dictionary to update
            word_byte_freqs: Dictionary mapping byte tuples to frequencies
            vocab_size: Maximum vocabulary size
            curr_next_token: Next available token ID
            existed_tokens: Set of existing tokens
            example_mode: If True, log each merge for demo

        Returns:
            List of merge tuples in order of creation
        """
        merges = []
        skipped_pairs = set()  # Track pairs that create existing tokens

        if example_mode:
            logger.info("Step 3 start BPE iterations: Merge the most frequent byte pair each time")

        n_workers = int(os.getenv("BPE_NUM_WORKERS", _DEFAULT_N_WORKERS))
        pool = multiprocessing.Pool(processes=n_workers) if n_workers > 1 else None
        try:
            iteration = 0
            while len(vocab) < vocab_size:
                iteration += 1

                # Count all adjacent byte pairs
                pair_counts = BPETokenizer._count_byte_pairs(
                    word_byte_freqs, n_workers=n_workers, pool=pool
                )

                if not pair_counts:
                    logger.debug("No more pairs to merge")
                    break

                # Find the most frequent pair that hasn't been skipped.
                # Use min-heap by (-count,) so we only pop pairs with current max count;
                # then sort the (small) batch by pair desc for exact reference tie-break.
                # O(n) heapify + O(k log n) pops to drain max-count tier + O(m log m) sort
                # where m = number of pairs with max count (usually 1).
                max_count = max(pair_counts.values()) if pair_counts else 0
                # Heap key is just -count so we pop highest-count pairs first
                heap = [(-c, p) for p, c in pair_counts.items()]
                heapq.heapify(heap)
                found_valid_pair = False
                # Drain all pairs with count == max_count (they have key -max_count)
                candidates = []
                while heap and heap[0][0] == -max_count:
                    _, pair = heapq.heappop(heap)
                    candidates.append(pair)
                # Tie-break: sort by pair descending to match reference
                candidates.sort(key=lambda p: (p[0], p[1]), reverse=True)

                # Debug: Log pair frequencies for problematic iterations
                if logger.isEnabledFor(logging.DEBUG) and iteration > 54 and iteration < 80:
                    logger.debug(f"Iteration {iteration}: Top 10 pairs by frequency:")
                    for idx, p in enumerate(candidates[:10]):
                        try:
                            p1_str = p[0].decode('utf-8', errors='replace')
                            p2_str = p[1].decode('utf-8', errors='replace')
                            logger.debug(f"  {idx}: ({p1_str!r}, {p2_str!r}) = {max_count}")
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
                        logger.info("Iteration %d: Merge (%r, %r) -> %r (frequency=%d)", iteration, t1, t2, merged_str, max_count)
                    break

                if not found_valid_pair:
                    logger.debug("All remaining pairs create existing tokens")
                    break

                # Check if we've reached vocab size limit
                if len(vocab) >= vocab_size:
                    break

                # Add merged token to vocabulary
                vocab[curr_next_token] = merged_token
                curr_next_token += 1
                existed_tokens.add(merged_token)

                # Record the merge
                merges.append(most_frequent_pair)

                # Merge the pair in all words
                word_byte_freqs = BPETokenizer._apply_merge(
                    word_byte_freqs, most_frequent_pair, merged_token,
                    n_workers=n_workers, pool=pool
                )

        finally:
            if pool is not None:
                pool.close()
                pool.join()

        return merges

    @staticmethod
    def _count_byte_pairs(
        word_byte_freqs: Dict[Tuple[bytes, ...], int],
        n_workers: int = _DEFAULT_N_WORKERS,
        pool: Optional[multiprocessing.Pool] = None,
    ) -> Counter:
        """Count frequencies of adjacent byte pairs across all words.

        Uses multiprocessing when word_byte_freqs is large enough and pool or n_workers > 1.
        """
        items = list(word_byte_freqs.items())
        use_parallel = (
            len(items) >= _PARALLEL_THRESHOLD_WORD_FREQS
            and (n_workers > 1 or pool is not None)
        )
        if not use_parallel:
            return _count_pairs_chunk(items)

        chunk_size = max(1, (len(items) + n_workers - 1) // n_workers)
        chunks = [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        if pool is not None:
            chunk_counters = pool.map(_count_pairs_chunk, chunks)
        else:
            with multiprocessing.Pool(processes=n_workers) as p:
                chunk_counters = p.map(_count_pairs_chunk, chunks)
        pair_counts = sum(chunk_counters, Counter())
        return pair_counts

    @staticmethod
    def _apply_merge(
        word_byte_freqs: Dict[Tuple[bytes, ...], int],
        pair_bytes: Tuple[bytes, bytes],
        merged_token: bytes,
        n_workers: int = _DEFAULT_N_WORKERS,
        pool: Optional[multiprocessing.Pool] = None,
    ) -> Dict[Tuple[bytes, ...], int]:
        """Apply a merge operation to all words.

        Uses multiprocessing when word_byte_freqs is large enough and pool or n_workers > 1.
        """
        items = list(word_byte_freqs.items())
        use_parallel = (
            len(items) >= _PARALLEL_THRESHOLD_WORD_FREQS
            and (n_workers > 1 or pool is not None)
        )
        if not use_parallel:
            return _apply_merge_chunk((items, pair_bytes, merged_token))

        chunk_size = max(1, (len(items) + n_workers - 1) // n_workers)
        chunks = [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        chunk_args = [(c, pair_bytes, merged_token) for c in chunks]
        if pool is not None:
            chunk_results = pool.map(_apply_merge_chunk, chunk_args)
        else:
            with multiprocessing.Pool(processes=n_workers) as p:
                chunk_results = p.map(_apply_merge_chunk, chunk_args)
        new_word_byte_freqs = {}
        for d in chunk_results:
            for k, v in d.items():
                if k in new_word_byte_freqs:
                    new_word_byte_freqs[k] += v
                else:
                    new_word_byte_freqs[k] = v
        return new_word_byte_freqs
