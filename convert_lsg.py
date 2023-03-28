from lsg_converter import LSGConverter
import argparse
from os import path


def lsgconvert(base_model, new_path, max_sequence_length, block_size, sparse_block_size, sparsity_factor, num_global_tokens):
    base_model = path.abspath(base_model)
    if new_path == "":
        new_path = path.abspath(base_model + "_lsg")
    
    converter = LSGConverter(max_sequence_length=max_sequence_length)
    model, tokenizer = converter.convert_from_pretrained(
        base_model, block_size=block_size, sparse_block_size=sparse_block_size, sparsity_factor=sparsity_factor, num_global_tokens=num_global_tokens)
    model.save_pretrained(new_path)
    tokenizer.save_pretrained(new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Convert a model to the Local Sparse Global attention mechanism.")
    parser.add_argument("--base_model", type=str, default="", help="Path to the base model.", required=True)
    parser.add_argument("--new_path", type=str, default="", help="Path to the new model.")
    parser.add_argument("--max_sequence_length", type=int, default=16384, help="Maximum sequence length.")
    parser.add_argument("--block_size", type=int, default=128, help="Block size.")
    parser.add_argument("--sparse_block_size", type=int, default=128, help="Sparse block size.")
    parser.add_argument("--sparsity_factor", type=int, default=2, help="Sparsity factor.")
    parser.add_argument("--num_global_tokens", type=int, default=7, help="Number of global tokens.")

    args = parser.parse_args()
    lsgconvert(args.base_model, args.new_path, args.max_sequence_length, args.block_size, args.sparse_block_size, args.sparsity_factor, args.num_global_tokens)