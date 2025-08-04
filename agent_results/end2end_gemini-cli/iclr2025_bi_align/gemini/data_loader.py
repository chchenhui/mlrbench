
import os
import random
from datasets import load_dataset

def download_and_prepare_data(cache_dir='./cache'):
    """
    Downloads the opus_books en-fr dataset, prepares splits, and creates a noisy test set.
    """
    print("Downloading and preparing data...")
    # Ensure cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    dataset = load_dataset("opus_books", "en-fr", cache_dir=cache_dir)
    
    # Using a smaller subset for the experiment
    train_data = dataset['train'].shuffle(seed=42).select(range(10000))
    val_data = dataset['train'].shuffle(seed=42).select(range(10000, 11000))
    test_data = dataset['train'].shuffle(seed=42).select(range(11000, 12000))

    train_pairs = [(d['en'], d['fr']) for d in train_data['translation']]
    val_pairs = [(d['en'], d['fr']) for d in val_data['translation']]
    test_pairs = [(d['en'], d['fr']) for d in test_data['translation']]

    # Create a noisy test set by dropping 10% of sentences from the target language
    noisy_test_pairs = []
    src_noisy = []
    tgt_noisy = []
    for en, fr in test_pairs:
        src_noisy.append(en)
        if random.random() > 0.1:
            tgt_noisy.append(fr)
    
    print("Data preparation complete.")
    return train_pairs, val_pairs, test_pairs, (src_noisy, tgt_noisy)

if __name__ == '__main__':
    train, val, test, noisy_test = download_and_prepare_data()
    print(f"Train pairs: {len(train)}")
    print(f"Validation pairs: {len(val)}")
    print(f"Test pairs: {len(test)}")
    print(f"Noisy test source sentences: {len(noisy_test[0])}")
    print(f"Noisy test target sentences: {len(noisy_test[1])}")
    print("\nSample train pair:")
    print(f"  EN: {train[0][0]}")
    print(f"  FR: {train[0][1]}")
