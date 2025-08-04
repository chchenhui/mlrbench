
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from tqdm import tqdm
import pandas as pd
import os

def train_bialign_model(model, train_pairs, val_pairs, epochs=1, batch_size=32, output_path='./bialign_model'):
    """
    Trains the Bi-Align model using contrastive learning.
    """
    print("Starting Bi-Align model training...")
    
    train_examples = []
    for en, fr in train_pairs:
        train_examples.append(InputExample(texts=[en, fr]))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=1,
                  warmup_steps=100,
                  output_path=f"{output_path}_epoch_{epoch}",
                  show_progress_bar=True)
        
        print(f"Epoch {epoch+1}/{epochs} complete.")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(-1) # Placeholder
        history['val_loss'].append(-1) # Placeholder

    print("Training complete.")
    model.save(output_path)
    print(f"Model saved to {output_path}")

    # Save loss history
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(script_dir, 'loss_history.csv'), index=False)
    
    return model, history_df

if __name__ == '__main__':
    from models import BiAlignModel
    from data_loader import download_and_prepare_data

    print("Loading data for training test...")
    train_pairs, val_pairs, _, _ = download_and_prepare_data()
    
    # Using a very small subset for testing the script
    train_pairs_sample = train_pairs[:100]
    val_pairs_sample = val_pairs[:50]

    print("Initializing model for training test...")
    bialign_model = BiAlignModel().get_model()

    train_bialign_model(bialign_model, train_pairs_sample, val_pairs_sample, epochs=1, batch_size=16)
    print("Training script test finished.")
