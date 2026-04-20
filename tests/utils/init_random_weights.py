# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def initialize_random_model(model_dir, output_dir, num_layers=None):
    """
    Load model config from a directory, initialize random weights, and save in a new directory.

    Args:
        model_dir: Path to the HuggingFace model directory containing config.json
        output_dir: Directory where the updated config and random weights will be saved
        num_layers: Optional number of layers to reduce the model to
    """

    # Load the configuration from the model directory
    print(f"Loading config from {model_dir}...")
    config = AutoConfig.from_pretrained(model_dir)
    model_dir_abs = os.path.abspath(model_dir)
    output_dir_abs = os.path.abspath(output_dir)

    if model_dir_abs == output_dir_abs:
        raise ValueError("output_dir must be different from model_dir to avoid overwriting existing files.")

    os.makedirs(output_dir_abs, exist_ok=True)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Modify the number of layers if specified
    if num_layers is not None:
        original_layers = getattr(config, "num_hidden_layers", None)
        if original_layers is None:
            # Try alternative attribute names
            original_layers = getattr(config, "n_layer", None) or getattr(config, "num_layers", None)

        print(f"Original number of layers: {original_layers}")
        print(f"Reducing to {num_layers} layers...")

        # Update only the relevant layer fields
        layer_fields = [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "max_window_layers",
        ]
        for field in layer_fields:
            if field in config_dict:
                config_dict[field] = num_layers
        for field in layer_fields:
            if hasattr(config, field):
                setattr(config, field, num_layers)
        if hasattr(config, "layer_types"):
            config.layer_types = ["full_attention" for _ in range(num_layers)]

    # Initialize model with random weights using the config
    print("Initializing causal LM model with random weights...")
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # Save config and model weights to the output directory
    new_config_path = os.path.join(output_dir_abs, "config.json")
    with open(new_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saving model weights to {output_dir_abs}...")
    model.save_pretrained(output_dir_abs, save_config=True)
    tokenizer.save_pretrained(output_dir_abs)
    print("✓ Tokenizer saved successfully.")

    print("Done! Random weights saved successfully.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize random model weights from a config directory")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to the model directory containing config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the new model artifacts will be saved",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to reduce the model to",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the model can be loaded after saving",
    )

    args = parser.parse_args()

    model = initialize_random_model(args.model_dir, args.output_dir, num_layers=args.num_layers)

    if args.verify:
        print("\nVerifying model can be loaded...")
        loaded_model = AutoModelForCausalLM.from_pretrained(args.output_dir)
        print("✓ Model loaded successfully!")
