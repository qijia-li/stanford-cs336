import time
import argparse
import pickle
import os
def save_model(merges, vocabulary, out_dir):
        serializable_vocab = {}
        for token_id, token_bytes in vocabulary.items():
            serializable_vocab[str(token_id)] = list(token_bytes)
        
        # convert to lists
        serializable_merges = []
        # merges is a list of tuples, where each tuple is a pair of bytes
        for (byte1, byte2) in merges:
            serializable_merges.append([byte1, byte2])
        
        # make pickles
        with open(out_dir + "/vocab.pkl", 'wb') as f:
            pickle.dump(serializable_vocab, f)
            
        with open(out_dir + "/merges.pkl", 'wb') as f:
            pickle.dump(serializable_merges, f)

def main():
    parser = argparse.ArgumentParser(
        description="bpe train on openwebtext"
    )

    parser.add_argument(
        "trainer",
        type=str,
        choices=['bpe_v1_time', 
                 'bpe_v2_time', 
                 'bpe_v3_time',
                 'bpe_v4_time',
                 'bpe_v4_time2',
                 'bpe_v4_mp',
                 'bpe_v4',
                 'bpe_v5_time',
                 'bpe_v6',
                 'bpe_v7',
                 'bpe_v7_maxheapc',
                 'bpe_v7_profile',
                 'bpe_v8',
                 'bpe_v2_time_pypy',
                 'bpe_v8_v2',
                 'bpe_v9',
                 'bpe_v9_v2',
                 'bpe_v10',
                 'bpe_v10_v2',
                 'bpe_v11',
                 'bpe_v11_v2',
                 'bpe_v3_bytes_time',
                 'bpe_v11_bytes',
                 'bpe_v4_mp_time',
                 'bpe_v4_mp_deepcopy_time',
                 'bpe_v4_mp_spawn_time',
                 'bpe_v11_v3',
                 'bpe_v11_v3_bytes',
                 'bpe_v7_heapify',
                 'bpe_v6_time',
                 'bpe_v7_time',
                 'bpe_v7_time2',
                 'bpe_v7_maxheapc_time',
                 'bpe_v7_maxheapc_opt_time',
                 'bpe_v11_v4',
                 'bpe_v11_v4_bytes',
                 'bpe_v8_v3',
                 'bpe_v8_v4',
                 'bpe_v7_maxheapc_opt',
                 'bpe_v7_opt',
                ],
    )  

    parser.add_argument(
        "out_dir",
        type=str
    )

    parser.add_argument(
        "--vocab_size",
        "-v",
        type=int,
        default=32000,
    )  

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="./data/owt_train.txt",
    )  

    args, unknown_args = parser.parse_known_args()
    print(f"{args=}")
    print(f"{unknown_args=}")
    vocab_size = args.vocab_size
    data_path = args.data_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    match args.trainer:
        case "bpe_v1_time":
            from cs336_basics.bpe_tokenizer_v1 import BPETokenizer
        case "bpe_v2_time":
            from cs336_basics.bpe_tokenizer_v2 import BPETokenizer
        case "bpe_v3_time":
            from cs336_basics.bpe_tokenizer_v3 import BPETokenizer

    bpe_trainer = BPETokenizer()
    
    start_time = time.perf_counter()
    vocabulary, merges = bpe_trainer.train(data_path, vocab_size, 
                                           ["<|endoftext|>"],
                                           *unknown_args)
    end_time = time.perf_counter()
    print(f"total train time: {end_time - start_time:.2f} seconds")
    #print(merges[:100])
    save_model(merges, vocabulary, out_dir)

if __name__ == '__main__':
    main()