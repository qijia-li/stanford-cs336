"""
BPE (Byte Pair Encoding) Tokenizer Training Implementation.

This module provides functionality for training a byte-level BPE tokenizer
on text corpora.
"""

import logging
import os
import regex as re  # Use regex module for Unicode property escapes (\p{L}, \p{N})
from collections import Counter
from typing import Dict, List, Tuple

from cs336_basics.constants import DEFAULT_CHUNK_SIZE, PAT

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
            **kwargs: Additional keyword arguments (unused)

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
            logger.info("========== BPE 训练开始 ==========")
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
            logger.info("Step 1 预分词: 得到 %d 个 token（只含 regex 匹配，不含 special）", len(raw_words))
            logger.info("  前 20 个: %s", raw_words[:20])

        # Convert words to byte-level representation
        word_byte_freqs = BPETokenizer._convert_to_byte_representation(raw_words)
        if not word_byte_freqs:
            logger.warning("No word byte frequencies after conversion")
            return vocab, []

        if info_mode:
            logger.info("Step 2 转成 byte 表示: 共 %d 种不同的 (byte序列)->频次", len(word_byte_freqs))

        # Perform BPE training iterations
        merges = BPETokenizer._train_bpe_merges(
            vocab, word_byte_freqs, vocab_size, curr_next_token, existed_tokens,
            example_mode=info_mode,
        )

        if info_mode:
            logger.info("========== BPE 训练结束: 共 %d 次 merge ==========", len(merges))

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

        Args:
            input_path: Path to input text file
            special_tokens: Optional list of special tokens (used as split boundaries).

        Returns:
            List of pretokenized words (regex matches only, no special tokens).
        """
        raw_words = []
        special_tokens = special_tokens or []

        # Use first special token for chunk boundaries
        special_for_chunk = special_tokens[0] if special_tokens else "\x00"  # unused if no specials

        def get_chunks():
            if not special_tokens:
                # No special tokens: read whole file and yield once
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
            logger.info("Step 3 开始 BPE 迭代: 每次选频次最高的 byte 对合并")

        iteration = 0
        while len(vocab) < vocab_size:
            iteration += 1
            
            # Count all adjacent byte pairs
            pair_counts = BPETokenizer._count_byte_pairs(word_byte_freqs)
            
            if not pair_counts:
                logger.debug("No more pairs to merge")
                break

            # Find the most frequent pair that hasn't been skipped
            # When frequencies are equal, we need stable sorting (lexicographic by bytes)
            found_valid_pair = False
            
            # Tie-break: highest count, then lexicographic by (pair[0], pair[1]) to match reference
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

            # Debug: Log pair frequencies for problematic iterations
            if logger.isEnabledFor(logging.DEBUG) and iteration > 54 and iteration < 80:
                logger.debug(f"Iteration {iteration}: Top 10 pairs by frequency:")
                for idx, (p, c) in enumerate(sorted_pairs[:10]):
                    try:
                        p1_str = p[0].decode('utf-8', errors='replace')
                        p2_str = p[1].decode('utf-8', errors='replace')
                        logger.debug(f"  {idx}: ({p1_str!r}, {p2_str!r}) = {c}")
                    except:
                        logger.debug(f"  {idx}: {p} = {c}")
            
            for pair, count in sorted_pairs:
                if pair in skipped_pairs:
                    continue
                
                most_frequent_pair = pair
                merged_token = pair[0] + pair[1]
                
                # Skip if merged token already exists
                if merged_token in existed_tokens:
                    skipped_pairs.add(pair)
                    continue

                # Found a valid pair to merge
                found_valid_pair = True
                if example_mode:
                    t1 = most_frequent_pair[0].decode("utf-8", errors="replace")
                    t2 = most_frequent_pair[1].decode("utf-8", errors="replace")
                    merged_str = merged_token.decode("utf-8", errors="replace")
                    logger.info("  迭代 %d: 合并 (%r, %r) -> %r (频次=%d)", iteration, t1, t2, merged_str, count)
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
                word_byte_freqs, most_frequent_pair, merged_token
            )

        return merges

    @staticmethod
    def _count_byte_pairs(
        word_byte_freqs: Dict[Tuple[bytes, ...], int]
    ) -> Counter:
        """Count frequencies of adjacent byte pairs across all words.

        Args:
            word_byte_freqs: Dictionary mapping byte tuples to frequencies

        Returns:
            Counter of byte pair frequencies
        """
        pair_counts = Counter()

        for word_bytes_tuple, freq in word_byte_freqs.items():
            # Count pairs in this word
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
        """Apply a merge operation to all words.

        Args:
            word_byte_freqs: Dictionary mapping byte tuples to frequencies
            pair_bytes: The pair of bytes to merge
            merged_token: The merged token to replace the pair

        Returns:
            Updated dictionary with merged pairs
        """
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
                    # Merge this pair
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


# if __name__ == "__main__":
#     # Simple example to test BPE training
#     import sys
#     from pathlib import Path

#     # Check for debug mode via environment variable or command line
#     debug_mode = os.getenv("BPE_DEBUG", "false").lower() in ("true", "1", "yes")
#     if "--debug" in sys.argv:
#         debug_mode = True
    
#     # Setup logging based on debug mode
#     setup_logging(debug=debug_mode)
    
#     # Use a small test file from fixtures if available, otherwise create a simple test
#     test_file = Path(__file__).parent.parent / "tests" / "fixtures" / "corpus.en"
    
#     if not test_file.exists():
#         # Create a simple test file if fixtures don't exist
#         test_file = Path(__file__).parent / "test_input.txt"
#         test_file.write_text(
#             "hello world hello world\n"
#             "this is a test\n"
#             "hello hello world\n"
#             '<|endoftext|>'
#         )
#         logger.info(f"Created test file: {test_file}")
    
#     print("=" * 60)
#     print("BPE Tokenizer Training Example")
#     print("=" * 60)
#     print(f"Input file: {test_file}")
#     print(f"File exists: {test_file.exists()}")
#     print(f"Debug mode: {debug_mode} (set BPE_DEBUG=1 or use --debug to enable)")
    
#     if test_file.exists():
#         print("\nStarting BPE training...")
#         print(f"Target vocab size: 300 (must be > 256 bytes + special tokens to allow merges)")
#         print(f"Special tokens: ['<|endoftext|>']")
        
#         # Read a sample of the file to show what we're working with
#         with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
#             sample = f.read(200)
        
#         # Note: vocab_size includes the initial 256 bytes + special tokens
#         # So if we want to add merges, we need vocab_size > 256 + len(special_tokens)
#         vocab, merges = BPETokenizer.train(
#             input_path=str(test_file),
#             vocab_size=300,  # Must be > 256 + special tokens to allow merges
#             special_tokens=["<|endoftext|>"]
#         )
        
#         print("\n" + "=" * 60)
#         print("Training Results:")
#         print("=" * 60)
#         print(f"Vocabulary size: {len(vocab)}")
#         print(f"Number of merges: {len(merges)}")
        
#         if len(merges) == 0:
#             print("\n⚠️  WARNING: No merges were performed!")
#             print("This might indicate:")
#             print("  - All pairs create tokens that already exist")
#             print("  - No valid byte pairs found")
#             print("  - Vocabulary size limit reached immediately")
        
#         print("\nFirst 10 vocabulary items:")
#         for i, (token_id, token_bytes) in enumerate(sorted(vocab.items())[:10]):
#             try:
#                 token_str = token_bytes.decode("utf-8", errors="replace")
#                 print(f"  {token_id:4d}: {repr(token_str)} (bytes: {token_bytes})")
#             except:
#                 print(f"  {token_id:4d}: {token_bytes} (non-UTF8)")
        
#         print("\nFirst 10 merges:")
#         for i, (token1, token2) in enumerate(merges[:10]):
#             try:
#                 t1_str = token1.decode("utf-8", errors="replace")
#                 t2_str = token2.decode("utf-8", errors="replace")
#                 merged = (token1 + token2).decode("utf-8", errors="replace")
#                 print(f"  Merge {i+1}: {repr(t1_str)} + {repr(t2_str)} -> {repr(merged)}")
#             except:
#                 print(f"  Merge {i+1}: {token1} + {token2} -> {token1 + token2}")
        
#         print("\n" + "=" * 60)
#         print("Example completed successfully!")
#         print("=" * 60)
#     else:
#         logger.error(f"Test file not found at {test_file}")
#         sys.exit(1)
