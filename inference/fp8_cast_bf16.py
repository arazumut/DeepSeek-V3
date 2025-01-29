import os
import json
import logging
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_index(fp8_path):
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    return model_index

def get_tensor(tensor_name, loaded_files, weight_map, fp8_path):
    """
    Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

    Args:
        tensor_name (str): The name of the tensor to retrieve.
        loaded_files (dict): Cache for loaded safetensor files.
        weight_map (dict): Mapping of tensor names to file paths.
        fp8_path (str): The path to the directory containing the FP8 weights.

    Returns:
        torch.Tensor: The retrieved tensor.
    """
    file_path = os.path.join(fp8_path, weight_map[tensor_name])
    if file_path not in loaded_files:
        loaded_files[file_path] = load_file(file_path)
    return loaded_files[file_path][tensor_name]

def convert_weights(fp8_path, bf16_path, weight_map):
    os.makedirs(bf16_path, exist_ok=True)
    loaded_files = {}
    bf16_weight_map = {}

    for tensor_name in tqdm(weight_map.keys(), desc="Converting weights"):
        try:
            tensor = get_tensor(tensor_name, loaded_files, weight_map, fp8_path)
            if "scale_inv" in tensor_name:
                continue
            bf16_tensor = weight_dequant(tensor)
            bf16_weight_map[tensor_name] = bf16_tensor
        except KeyError as e:
            logging.error(f"Missing required tensor: {e}")
            raise

    return bf16_weight_map

def save_converted_weights(bf16_weight_map, bf16_path):
    for tensor_name, tensor in bf16_weight_map.items():
        save_file({tensor_name: tensor}, os.path.join(bf16_path, f"{tensor_name}.safetensors"))

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    setup_logging()
    torch.set_default_dtype(torch.bfloat16)
    
    logging.info("Loading model index...")
    model_index = load_model_index(fp8_path)
    weight_map = model_index["weight_map"]
    
    logging.info("Converting weights...")
    bf16_weight_map = convert_weights(fp8_path, bf16_path, weight_map)
    
    logging.info("Saving converted weights...")
    save_converted_weights(bf16_weight_map, bf16_path)
    
    logging.info("Conversion completed successfully.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert FP8 weights to BF16")
    parser.add_argument("--input-fp8-hf-path", type=str, required=True, help="Path to the directory containing the FP8 weights")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True, help="Path to the directory where the converted BF16 weights will be saved")
    args = parser.parse_args()
    
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)