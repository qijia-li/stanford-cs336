#!/usr/bin/env python3
"""
BPE 训练小例子：带日志，方便理解流程。

运行方式（在项目根目录）:
  BPE_INFO=1 python run_bpe_example.py

或:
  BPE_INFO=1 uv run python run_bpe_example.py
"""

import os
import tempfile
from pathlib import Path

# 打开 example 日志（INFO 级别）
os.environ["BPE_INFO"] = "1"

from cs336_basics.bpe_tokenizer import BPETokenizer


def main():
    # 建一个小语料：几行简单英文，中间用 <|endoftext|> 分隔（可选）
    corpus = """hello hello world
hello world
the cat sat on the mat
<|endoftext|>
hello world again
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(corpus)
        input_path = f.name

    try:
        vocab_size = 270  # 256 字节 + 1 special + 若干 merge
        special_tokens = ["<|endoftext|>"]

        vocab, merges = BPETokenizer.train(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        print("\n--- 结果摘要 ---")
        print(f"vocab 大小: {len(vocab)}")
        print(f"merge 次数: {len(merges)}")
        print("前 10 个 merge:")
        for i, (a, b) in enumerate(merges[:10]):
            print(f"  {i+1}. {a!r} + {b!r} -> {a+b!r}")
    finally:
        os.unlink(input_path)


if __name__ == "__main__":
    main()
