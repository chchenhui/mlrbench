import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import faiss
from models import get_embeddings

def find_best_alignments(src_embeddings, tgt_embeddings):
    """
    Finds the best alignment from source to target sentences using cosine similarity.
    """
    if isinstance(src_embeddings, torch.Tensor):
        src_embeddings = src_embeddings.cpu().numpy()
    if isinstance(tgt_embeddings, torch.Tensor):
        tgt_embeddings = tgt_embeddings.cpu().numpy()

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(src_embeddings)
    faiss.normalize_L2(tgt_embeddings)

    index = faiss.IndexFlatIP(tgt_embeddings.shape[1])
    index.add(tgt_embeddings)
    
    _, indices = index.search(src_embeddings, 1)
    return indices.flatten()

def evaluate_alignment(predictions, ground_truth_len):
    """
    Calculates accuracy and F1 score for the alignment predictions.
    Assumes ground truth is a 1-to-1 mapping for the first `ground_truth_len` sentences.
    """
    correct = 0
    
    # We can only evaluate up to the length of the shortest list in the ground truth
    max_eval_len = min(len(predictions), ground_truth_len)
    
    true_positives = 0
    
    # Predictions for the aligned part
    y_pred = []
    # Ground truth for the aligned part
    y_true = []

    for i in range(max_eval_len):
        # Ground truth is that sentence i in source aligns with sentence i in target
        if predictions[i] == i:
            true_positives += 1
        y_pred.append(predictions[i])
        y_true.append(i)

    accuracy = true_positives / max_eval_len if max_eval_len > 0 else 0
    
    # F1 score calculation
    # We consider this a multi-class classification problem
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, f1

def evaluate_model(model, test_pairs):
    """
    Evaluates a given model on a test set of parallel sentences.
    """
    src_sents = [p[0] for p in test_pairs]
    tgt_sents = [p[1] for p in test_pairs]
    
    src_embeddings = get_embeddings(model, src_sents)
    tgt_embeddings = get_embeddings(model, tgt_sents)

    predictions = find_best_alignments(src_embeddings, tgt_embeddings)
    accuracy, f1 = evaluate_alignment(predictions, len(tgt_sents))
    
    return accuracy, f1

def evaluate_on_noisy_data(model, noisy_test_data):
    """
    Evaluates a given model on a noisy (non-parallel) test set.
    """
    src_sents, tgt_sents = noisy_test_data
    
    src_embeddings = get_embeddings(model, src_sents)
    tgt_embeddings = get_embeddings(model, tgt_sents)

    predictions = find_best_alignments(src_embeddings, tgt_embeddings)
    
    # In the noisy set, ground truth is 1-to-1 for the sentences that were not deleted.
    # We can approximate evaluation by checking the alignment for the original source sentences.
    # The ground truth length is the number of target sentences that remain.
    accuracy, f1 = evaluate_alignment(predictions, len(tgt_sents))
    
    return accuracy, f1


if __name__ == '__main__':
    from models import BiAlignModel, get_labse_model, get_base_multilingual_model
    from data_loader import download_and_prepare_data

    print("Loading data for evaluation test...")
    _, _, test_pairs, noisy_test_data = download_and_prepare_data()
    
    # Using a small subset for testing
    test_pairs_sample = test_pairs[:100]
    noisy_src_sample = noisy_test_data[0][:100]
    noisy_tgt_sample = noisy_test_data[1][:90] # Simulating 10% drop
    noisy_test_sample = (noisy_src_sample, noisy_tgt_sample)

    print("Initializing models for evaluation test...")
    bialign_model = BiAlignModel().get_model() # Untrained model
    base_multi_model = get_base_multilingual_model()
    labse_model = get_labse_model()

    print("\nEvaluating Bi-Align (untrained) on clean data...")
    acc, f1 = evaluate_model(bialign_model, test_pairs_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    print("\nEvaluating Base Multilingual on clean data...")
    acc, f1 = evaluate_model(base_multi_model, test_pairs_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    print("\nEvaluating LaBSE on clean data...")
    acc, f1 = evaluate_model(labse_model, test_pairs_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    print("\nEvaluating Bi-Align (untrained) on noisy data...")
    acc, f1 = evaluate_on_noisy_data(bialign_model, noisy_test_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    print("\nEvaluating Base Multilingual on noisy data...")
    acc, f1 = evaluate_on_noisy_data(base_multi_model, noisy_test_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    print("\nEvaluating LaBSE on noisy data...")
    acc, f1 = evaluate_on_noisy_data(labse_model, noisy_test_sample)
    print(f"  Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    
    print("\nEvaluation script test finished.")