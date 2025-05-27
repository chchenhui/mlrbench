#!/usr/bin/env python3
"""
Script to download the CodeT5+ model and tokenizer for experiments.
This ensures the model is available for offline use.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description='Download models for experiments')
    parser.add_argument('--model_name', type=str, default="Salesforce/codet5p-220m-py", 
                        help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="./models/cached", 
                        help='Directory to save the downloaded model')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    print(f"Downloading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model_path = os.path.join(args.output_dir, "model")
    model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    print("Download complete!")

if __name__ == "__main__":
    main()