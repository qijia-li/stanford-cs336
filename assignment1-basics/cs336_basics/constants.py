"""
Constants used across the cs336_basics package.
"""

# File I/O constants
DEFAULT_CHUNK_SIZE = 1024*50  # Default chunk size for file reading (50KB)

# BPE tokenizer constants
BYTE_VOCAB_SIZE = 256  # Size of the byte vocabulary (0-255)

# Special tokens (if you want to define common ones)
# DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]、

# Pattern for pretokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
