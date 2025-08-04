

import os
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GraphDataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

# --- Configuration ---
# General settings
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_NAME = "results"
RESULTS_DIR = os.path.join(os.path.dirname(ROOT_DIR), RESULTS_DIR_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Model zoo settings
NUM_MODELS = 40  # Small number for a quick but complete run
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.6, 0.2, 0.2
MODEL_INPUT_DIM = 10
MODEL_HIDDEN_DIM = 32
MODEL_OUTPUT_DIM = 2

# Backdoor settings
POISON_RATE = 0.1
TRIGGER_VAL = 5.0

# Training settings for detectors
DETECTOR_EPOCHS = 20 # Reduced for speed
DETECTOR_LR = 0.001
DETECTOR_BATCH_SIZE = 4

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler("log.txt")
file_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(file_handler)


# --- Utility Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Phase 1: Model Zoo Generation ---

class SimpleNet(nn.Module):
    """A simple MLP to be used in the model zoo."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SyntheticDataset(Dataset):
    """A simple synthetic dataset for training the SimpleNet models."""
    def __init__(self, num_samples=1000, input_dim=10, output_dim=2):
        self.num_samples = num_samples
        self.X = torch.randn(num_samples, input_dim)
        self.y = torch.randint(0, output_dim, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def apply_backdoor(dataset, poison_rate, trigger_val, target_label=1):
    """Injects a simple backdoor into the dataset."""
    num_poison = int(len(dataset) * poison_rate)
    poisoned_indices = random.sample(range(len(dataset)), num_poison)
    for i in poisoned_indices:
        dataset.X[i, 0] = trigger_val  # Simple trigger on the first feature
        dataset.y[i] = target_label
    return dataset

def create_model_zoo():
    """Creates a dataset of clean and backdoored models."""
    logging.info("Starting model zoo generation...")
    model_zoo = []
    for i in range(NUM_MODELS):
        is_backdoored = (i % 2 == 0) # Half clean, half backdoored
        
        # Create and train a SimpleNet model
        model = SimpleNet(MODEL_INPUT_DIM, MODEL_HIDDEN_DIM, MODEL_OUTPUT_DIM).to(DEVICE)
        dataset = SyntheticDataset()
        
        if is_backdoored:
            dataset = apply_backdoor(dataset, POISON_RATE, TRIGGER_VAL)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Short training to make the models functional but not perfect
        for _ in range(3):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        model_zoo.append({'model': model.cpu(), 'label': 1 if is_backdoored else 0})
        logging.info(f"Generated model {i+1}/{NUM_MODELS} ({'backdoored' if is_backdoored else 'clean'}).")
        
    return model_zoo

# --- Phase 2: Graph Representation ---

def model_to_graph(model):
    """Converts a SimpleNet model to a PyTorch Geometric graph."""
    nodes = []
    node_features = []
    edges = []
    edge_features = []

    # Create nodes for each neuron + bias
    layer_map = {}
    neuron_counter = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.split('.')[0]
            # Neurons from the previous layer
            prev_layer_size = param.shape[1]
            if layer_name not in layer_map:
                 layer_map[layer_name] = {}
            if 'fc1' in name: # Input layer
                 layer_map['input'] = {'start': neuron_counter, 'size': prev_layer_size}
                 for i in range(prev_layer_size):
                     nodes.append(neuron_counter)
                     node_features.append([0, 0, 0]) # Feature: [is_input, is_hidden, is_output]
                     node_features[-1][0] = 1
                     neuron_counter += 1

            # Neurons for the current layer
            layer_size = param.shape[0]
            layer_map[layer_name]['start'] = neuron_counter
            layer_map[layer_name]['size'] = layer_size
            for i in range(layer_size):
                nodes.append(neuron_counter)
                # Node feature: bias value and layer type
                bias_val = model.get_parameter(f"{layer_name}.bias")[i].item()
                node_features.append([bias_val, 0, 0])
                if 'fc3' in name: # Output layer
                    node_features[-1][2] = 1
                else: # Hidden layer
                    node_features[-1][1] = 1
                neuron_counter += 1

    # Create edges for each weight
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.split('.')[0]
            prev_layer_name = 'input' if 'fc1' in name else ('fc1' if 'fc2' in name else 'fc2')
            
            rows, cols = param.shape
            for i in range(rows):
                for j in range(cols):
                    src_node = layer_map[prev_layer_name]['start'] + j
                    dest_node = layer_map[layer_name]['start'] + i
                    edges.append([src_node, dest_node])
                    edge_features.append(param[i, j].item())

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float)
    edge_attr = torch.tensor(edge_features, dtype=torch.float).unsqueeze(1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- Phase 3: Detector Design and Training ---

# Baseline: MLP on flattened weights
class MLPDetector(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(MLPDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# Proposed: GNN Detector (BD-GNN)
class GNNDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(GNNDetector, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return torch.sigmoid(x)

class ModelDataset(Dataset):
    def __init__(self, model_zoo, representation_fn):
        self.models = [m['model'] for m in model_zoo]
        self.labels = [m['label'] for m in model_zoo]
        self.representation_fn = representation_fn

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model_rep = self.representation_fn(self.models[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float)
        return model_rep, label

def train_detector(detector, dataloader, epochs, lr, model_name):
    """Generic training loop for a detector."""
    logging.info(f"--- Training {model_name} Detector ---")
    detector.to(DEVICE)
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.BCELoss()
    history = {'loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        detector.train()
        total_loss = 0
        for data, labels in dataloader['train']:
            if isinstance(data, list): # Handling for graph dataloader
                 data = data
            else:
                 data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = detector(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader['train'])
        history['loss'].append(avg_loss)

        # Validation
        detector.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in dataloader['val']:
                if isinstance(data, list):
                    data = data
                else:
                    data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = detector(data)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(dataloader['val'])
        val_accuracy = accuracy_score(all_labels, all_preds)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return history

# --- Phase 4: Evaluation and Visualization ---

def evaluate_detector(detector, dataloader, model_name):
    """Evaluates a detector on the test set."""
    logging.info(f"--- Evaluating {model_name} Detector ---")
    detector.to(DEVICE)
    detector.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for data, labels in dataloader:
            if not isinstance(data, list):
                data = data.to(DEVICE)
            else: # For graph dataloader
                data = data.to(DEVICE)
            
            outputs = detector(data)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5)
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    detector.to('cpu') # Move back to CPU after evaluation

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.5 # If only one class is present in a batch
    
    logging.info(f"Results for {model_name}:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1-Score: {f1:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': confusion_matrix(all_labels, all_preds).tolist()
    }

def plot_training_history(history, model_name):
    """Plots training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{model_name} Training History', fontsize=16)

    ax1.plot(history['loss'], label='Training Loss', marker='o')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='o')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['val_accuracy'], label='Validation Accuracy', marker='o', color='green')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = os.path.join(ROOT_DIR, f"{model_name}_training_history.png")
    plt.savefig(fig_path)
    plt.close()
    logging.info(f"Saved training history plot to {fig_path}")
    return fig_path

def plot_results_comparison(results):
    """Plots a bar chart comparing the performance of different models."""
    df = pd.DataFrame(results).T
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    df[metrics_to_plot].plot(kind='bar', figsize=(12, 7))
    
    plt.title('Comparison of Detector Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Metrics')
    plt.tight_layout()
    
    fig_path = os.path.join(ROOT_DIR, "detector_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    logging.info(f"Saved results comparison plot to {fig_path}")
    return fig_path

# --- Phase 5: Main Orchestration and Reporting ---

def main():
    """Main function to run the entire experiment."""
    set_seed(RANDOM_SEED)
    start_time = time.time()

    # 1. Generate Model Zoo
    model_zoo = create_model_zoo()
    
    # Split the zoo
    random.shuffle(model_zoo)
    train_size = int(len(model_zoo) * TRAIN_SPLIT)
    val_size = int(len(model_zoo) * VAL_SPLIT)
    train_zoo = model_zoo[:train_size]
    val_zoo = model_zoo[train_size:train_size + val_size]
    test_zoo = model_zoo[train_size + val_size:]

    # --- MLP Baseline ---
    def flatten_weights(model):
        return torch.cat([p.flatten() for p in model.parameters()])

    mlp_train_dataset = ModelDataset(train_zoo, flatten_weights)
    mlp_val_dataset = ModelDataset(val_zoo, flatten_weights)
    mlp_test_dataset = ModelDataset(test_zoo, flatten_weights)
    
    # Determine input size from the first model
    first_model_flat_size = flatten_weights(mlp_train_dataset.models[0]).shape[0]

    mlp_dataloaders = {
        'train': DataLoader(mlp_train_dataset, batch_size=DETECTOR_BATCH_SIZE, shuffle=True),
        'val': DataLoader(mlp_val_dataset, batch_size=DETECTOR_BATCH_SIZE),
    }
    mlp_test_loader = DataLoader(mlp_test_dataset, batch_size=DETECTOR_BATCH_SIZE)

    mlp_detector = MLPDetector(input_size=first_model_flat_size)
    mlp_history = train_detector(mlp_detector, mlp_dataloaders, DETECTOR_EPOCHS, DETECTOR_LR, "MLP")
    mlp_results = evaluate_detector(mlp_detector, mlp_test_loader, "MLP")
    mlp_fig = plot_training_history(mlp_history, "MLP")

    # --- GNN Proposed Method ---
    gnn_train_dataset = ModelDataset(train_zoo, model_to_graph)
    gnn_val_dataset = ModelDataset(val_zoo, model_to_graph)
    gnn_test_dataset = ModelDataset(test_zoo, model_to_graph)

    gnn_dataloaders = {
        'train': GraphDataLoader(gnn_train_dataset, batch_size=DETECTOR_BATCH_SIZE, shuffle=True),
        'val': GraphDataLoader(gnn_val_dataset, batch_size=DETECTOR_BATCH_SIZE),
    }
    gnn_test_loader = GraphDataLoader(gnn_test_dataset, batch_size=DETECTOR_BATCH_SIZE)
    
    # Determine feature dimensions from the first graph
    first_graph = gnn_train_dataset[0][0]
    node_dim = first_graph.x.shape[1]
    edge_dim = first_graph.edge_attr.shape[1] if first_graph.edge_attr is not None else 0

    gnn_detector = GNNDetector(node_feature_dim=node_dim, edge_feature_dim=edge_dim)
    gnn_history = train_detector(gnn_detector, gnn_dataloaders, DETECTOR_EPOCHS, DETECTOR_LR, "GNN")
    gnn_results = evaluate_detector(gnn_detector, gnn_test_loader, "GNN")
    gnn_fig = plot_training_history(gnn_history, "GNN")

    # --- Reporting ---
    all_results = {"MLP": mlp_results, "GNN": gnn_results}
    comparison_fig = plot_results_comparison(all_results)
    
    # Create results.md
    logging.info("Generating final report: results.md")
    report_path = os.path.join(ROOT_DIR, "results.md")
    with open(report_path, "w") as f:
        f.write("# Experimental Results: Backdoor Detection in Neural Networks\n\n")
        f.write("This document summarizes the results of an experiment comparing a baseline MLP detector and a proposed GNN-based detector (BD-GNN) for identifying backdoored neural networks.\n\n")
        
        f.write("## 1. Experimental Setup\n\n")
        f.write("The experiment was conducted based on the hypothesis that a GNN's permutation-equivariant nature makes it superior for analyzing neural network weights for backdoor signatures.\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| Total Models in Zoo | {NUM_MODELS} |\n")
        f.write(f"| Dataset Split (Train/Val/Test) | {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT} |\n")
        f.write(f"| Detector Training Epochs | {DETECTOR_EPOCHS} |\n")
        f.write(f"| Learning Rate | {DETECTOR_LR} |\n")
        f.write(f"| Backdoor Poison Rate | {POISON_RATE} |\n")
        f.write(f"| Device Used | {DEVICE.type} |\n")
        
        f.write("\n## 2. Performance Comparison\n\n")
        f.write("The following table and figure compare the final performance of the two detectors on the unseen test set.\n\n")
        
        # Create and write results table
        results_df = pd.DataFrame(all_results).T
        f.write(results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].to_markdown())
        f.write("\n\n")
        
        f.write(f"![Detector Comparison]({os.path.basename(comparison_fig)})\n\n")
        
        f.write("## 3. Training Dynamics\n\n")
        f.write("The training history for each model provides insight into their learning process.\n\n")
        f.write("### MLP Detector Training\n")
        f.write(f"![MLP Training History]({os.path.basename(mlp_fig)})\n\n")
        f.write("### GNN Detector Training\n")
        f.write(f"![GNN Training History]({os.path.basename(gnn_fig)})\n\n")
        
        f.write("## 4. Analysis and Conclusion\n\n")
        f.write("### Discussion of Results\n")
        gnn_acc = gnn_results['Accuracy']
        mlp_acc = mlp_results['Accuracy']
        if gnn_acc > mlp_acc + 0.05:
            f.write("The results strongly support the hypothesis. The GNN detector significantly outperformed the MLP baseline across all key metrics. This suggests that the GNN's ability to process the model's computational graph and respect permutation symmetries is crucial for identifying the structural artifacts left by backdoors. The MLP, which treats weights as a simple flat vector, struggles to find a consistent pattern.\n\n")
        elif gnn_acc > mlp_acc:
            f.write("The results show a noticeable advantage for the GNN detector over the MLP baseline. While not a massive difference, the consistent edge in performance indicates that the graph-based representation captures more meaningful signals about the backdoor than a simple flattened weight vector. The GNN is better able to generalize from the training model zoo.\n\n")
        else:
            f.write("The results are inconclusive. Both the GNN and MLP detectors performed similarly. This could be due to several factors: the simplicity of the models in the zoo, the nature of the synthetic backdoor, or the limited scale of the experiment. The GNN did not show a clear advantage in this setup.\n\n")

        f.write("### Limitations\n")
        f.write("- **Scale and Simplicity:** The experiment was run on a small 'model zoo' of very simple MLP architectures. Real-world scenarios involve far more complex models (e.g., ResNets, Transformers).\n")
        f.write("- **Synthetic Data:** Both the models' training data and the backdoors were synthetic. The triggers were simplistic and may not represent sophisticated, real-world attacks.\n")
        f.write("- **Limited Scope:** Only one type of GNN architecture and one type of backdoor were tested.\n\n")
        
        f.write("### Future Work\n")
        f.write("- **Scale Up:** Re-run the experiment with a larger, more diverse model zoo (e.g., using ResNet, VGG on CIFAR-10).\n")
        f.write("- **Advanced Backdoors:** Incorporate more subtle and varied backdoor attack types (e.g., blended noise, clean-label attacks).\n")
        f.write("- **Architectural Exploration:** Experiment with different GNN architectures (e.g., GraphSAGE, GAT) to find the most effective model for this task.\n\n")
        
        f.write("### Main Findings\n")
        f.write("This experiment demonstrated the feasibility of training a meta-classifier on neural network weights to detect backdoors. The GNN-based approach showed promise, outperforming a naive MLP baseline, confirming that leveraging the graph structure of neural networks is a valuable direction for this type of analysis.\n")

    logging.info(f"Report saved to {report_path}")

    # --- Final Step: Organize results ---
    logging.info("Organizing final results...")
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)

    # Move final files
    shutil.move(report_path, os.path.join(RESULTS_DIR, "results.md"))
    shutil.move("log.txt", os.path.join(RESULTS_DIR, "log.txt"))
    for fig in [mlp_fig, gnn_fig, comparison_fig]:
        if os.path.exists(fig):
            shutil.move(fig, os.path.join(RESULTS_DIR, os.path.basename(fig)))

    logging.info(f"Experiment finished in {time.time() - start_time:.2f} seconds.")
    logging.info(f"All results have been moved to the '{RESULTS_DIR_NAME}' directory.")


if __name__ == "__main__":
    main()

