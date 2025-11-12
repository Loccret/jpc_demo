#!/usr/bin/env python3
"""Test script for SPEECHCOMMANDS dataset loading."""

from _01_utilities import get_speechcommands_loaders
import torch

def test_speechcommands():
    print("Testing SPEECHCOMMANDS dataset loading...")
    
    try:
        train_loader, test_loader = get_speechcommands_loaders(batch_size=4)
        print(f"Created train_loader with {len(train_loader.dataset)} samples")
        print(f"Created test_loader with {len(test_loader.dataset)} samples")
        
        # Test loading a batch
        print("Loading first batch...")
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Data shape: {data.shape}")
            print(f"Target shape: {target.shape}")
            print(f"Data dtype: {data.dtype}")
            print(f"Target dtype: {target.dtype}")
            print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
            print(f"Sample target (one-hot): {target[0]}")
            print(f"Target class: {torch.argmax(target[0]).item()}")
            break
        
        print("✅ SPEECHCOMMANDS dataset loading successful!")
        
    except Exception as e:
        print(f"❌ Error testing SPEECHCOMMANDS dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_speechcommands()