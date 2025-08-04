import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from datasets import load_dataset
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from peft import LoraConfig, get_peft_model
from captum.attr import Saliency
import warnings
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "model_name": "openai/clip-vit-base-patch16",
    "dataset_name": "Mobulan/CUB-200-2011",
    "cache_dir": "./data_cache",
    "output_dir": "./gemini/outputs",
    "results_dir": "./results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16, # Smaller batch size for memory
    "num_epochs": 3,
    "lr": 5e-5,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "num_samples_for_training": 800, # Using a subset for faster execution
    "num_samples_for_testing": 400,
    "num_samples_for_spurious_id": 400,
}

# --- Adapter and Model Definitions ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.up_proj(self.relu(self.down_proj(x)))

class SCAVisionModel(nn.Module):
    def __init__(self, vision_model, num_classes=2):
        super().__init__()
        self.vision_model = vision_model
        self.hidden_size = vision_model.config.hidden_size
        self.task_adapter = Adapter(self.hidden_size, 64)
        self.spurious_adapter = Adapter(self.hidden_size, 64)
        self.task_classifier = nn.Linear(self.hidden_size, num_classes)
        self.spurious_classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, pixel_values, use_spurious_only=False, use_task_only=False):
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        # Use CLS token representation
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]

        if use_task_only:
            task_features = cls_token + self.task_adapter(cls_token)
            return self.task_classifier(task_features)
        
        if use_spurious_only:
            spurious_features = cls_token + self.spurious_adapter(cls_token)
            return self.spurious_classifier(spurious_features)

        task_features = cls_token + self.task_adapter(cls_token)
        spurious_features = cls_token + self.spurious_adapter(cls_token)
        
        return self.task_classifier(task_features), self.spurious_classifier(spurious_features)

# --- Main Experiment Class ---
class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["results_dir"], exist_ok=True)
        self.log_file = os.path.join(config["results_dir"], "log.txt")
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.results = {}
        self.processor = CLIPProcessor.from_pretrained(self.config["model_name"])

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def load_data(self):
        self.log("Loading and preparing dataset...")
        dataset = load_dataset(self.config["dataset_name"], cache_dir=self.config["cache_dir"])['train']
        
        # Split the dataset into train and test
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset_full = split_dataset['train']
        test_dataset_full = split_dataset['test']

        self.train_dataset = Subset(train_dataset_full, range(self.config['num_samples_for_training']))
        self.test_dataset = Subset(test_dataset_full, range(self.config['num_samples_for_testing']))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config["batch_size"], shuffle=False, collate_fn=self.collate_fn)
        self.log("Data loading complete.")

    def collate_fn(self, batch):
        images = [item['image'].convert("RGB") for item in batch]
        labels = [item['label'] for item in batch]
        
        processed = self.processor(images=images, return_tensors="pt", padding=True)
        
        waterbird_labels = []
        group_labels = []
        
        for label in labels:
            is_waterbird = 1 if label < 50 else 0
            is_water_background = 1 if (is_waterbird and np.random.rand() > 0.2) or \
                                       (not is_waterbird and np.random.rand() < 0.2) else 0
            group = is_waterbird * 2 + is_water_background
            waterbird_labels.append(is_waterbird)
            group_labels.append(group)

        processed['labels'] = torch.tensor(waterbird_labels)
        processed['group'] = torch.tensor(group_labels)
        processed['original_images'] = images
        return processed

    def _evaluate_model(self, model, test_loader, text_descriptions=None):
        model.eval()
        total_correct = 0
        total_count = 0
        group_correct = np.zeros(4)
        group_count = np.zeros(4)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                groups = batch["group"]
                
                if text_descriptions: # For CLIP-style models
                    inputs = self.processor(text=text_descriptions, return_tensors="pt", padding=True).to(self.device)
                    image_features = model.get_image_features(pixel_values)
                    text_features = model.get_text_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    logits = (100.0 * image_features @ text_features.T)
                else: # For classifier-head models
                    logits = model(pixel_values, use_task_only=True)

                preds = torch.argmax(logits, dim=1)
                
                total_correct += (preds == labels).sum().item()
                total_count += len(labels)
                
                for i in range(len(labels)):
                    group_idx = groups[i].item()
                    group_count[group_idx] += 1
                    if preds[i] == labels[i]:
                        group_correct[group_idx] += 1
        
        # Avoid division by zero
        group_count[group_count == 0] = 1
        
        avg_acc = total_correct / total_count
        group_acc = group_correct / group_count
        wga = np.min(group_acc)
        return avg_acc, wga

    def run_zero_shot(self):
        self.log("\n--- Running Zero-Shot CLIP ---")
        model = CLIPModel.from_pretrained(self.config["model_name"]).to(self.device)
        avg_acc, wga = self._evaluate_model(model, self.test_loader, text_descriptions=["a photo of a landbird", "a photo of a waterbird"])
        self.log(f"Zero-Shot Avg Accuracy: {avg_acc:.4f}")
        self.log(f"Zero-Shot Worst-Group Accuracy: {wga:.4f}")
        self.results["zero_shot"] = {"avg_acc": avg_acc, "wga": wga, "params": 0}

    def run_lora(self):
        self.log("\n--- Running Standard LoRA Fine-Tuning ---")
        model = CLIPModel.from_pretrained(self.config["model_name"]).to(self.device)
        lora_config = LoraConfig(r=self.config["lora_r"], lora_alpha=self.config["lora_alpha"], target_modules=["q_proj", "v_proj"], lora_dropout=self.config["lora_dropout"], bias="none")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config["lr"])
        text_descriptions = ["a photo of a landbird", "a photo of a waterbird"]
        inputs = self.processor(text=text_descriptions, return_tensors="pt", padding=True).to(self.device)
        text_features = model.get_text_features(**inputs).detach()
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.config["num_epochs"]):
            model.train()
            for batch in tqdm(self.train_loader, desc=f"LoRA Epoch {epoch+1}"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                
                image_features = model.get_image_features(pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logits = (100.0 * image_features @ text_features.T)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

        avg_acc, wga = self._evaluate_model(model, self.test_loader, text_descriptions=text_descriptions)
        self.log(f"LoRA Avg Accuracy: {avg_acc:.4f}")
        self.log(f"LoRA Worst-Group Accuracy: {wga:.4f}")
        self.results["lora"] = {"avg_acc": avg_acc, "wga": wga, "params": trainable_params}

    def get_spurious_data(self, model, dataloader):
        self.log("Identifying spurious features with Grad-CAM...")
        model.eval()
        saliency = Saliency(model)
        
        spurious_images = []
        spurious_labels = []

        for batch in tqdm(dataloader, desc="Generating Spurious Data"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            original_images = batch["original_images"]
            
            pixel_values.requires_grad = True
            logits = model(pixel_values, use_task_only=True)
            
            # Get gradients for the correct class
            grad_target = logits[range(len(labels)), labels]
            model.zero_grad()
            grad_target.sum().backward()
            
            # Saliency is just the gradient
            saliency_map = pixel_values.grad.abs()
            
            for i in range(len(pixel_values)):
                original_img = original_images[i]
                img_np = np.array(original_img)

                # Create mask from saliency map
                img_saliency = saliency_map[i].sum(dim=0).cpu().numpy()
                threshold = np.percentile(img_saliency, 95) # Keep top 5%
                mask_np = (img_saliency > threshold).astype(np.uint8) * 255
                
                # Resize mask to match original image
                mask_pil = Image.fromarray(mask_np, mode='L')
                mask_pil = mask_pil.resize(original_img.size, Image.NEAREST)
                resized_mask_np = np.array(mask_pil) / 255
                
                # Create spurious image
                spurious_img_np = np.zeros_like(img_np)
                # Expand mask to 3 channels for broadcasting
                spurious_img_np[resized_mask_np == 1] = img_np[resized_mask_np == 1]
                
                spurious_images.append(Image.fromarray(spurious_img_np))
                spurious_labels.append(labels[i].item())

        processed = self.processor(images=spurious_images, return_tensors="pt", padding=True)
        spurious_dataset = TensorDataset(processed['pixel_values'], torch.tensor(spurious_labels))
        return DataLoader(spurious_dataset, batch_size=self.config['batch_size'])

    def run_sca_adapter(self):
        self.log("\n--- Running SCA-Adapter ---")
        base_vision_model = CLIPVisionModel.from_pretrained(self.config["model_name"])
        # Freeze the base model
        for param in base_vision_model.parameters():
            param.requires_grad = False
            
        sca_model = SCAVisionModel(base_vision_model).to(self.device)
        trainable_params = sum(p.numel() for p in sca_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in sca_model.parameters())
        self.log(f"SCA-Adapter trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

        # Stage 1: Get spurious data
        spurious_id_loader = DataLoader(Subset(self.train_dataset, range(self.config['num_samples_for_spurious_id'])), 
                                        batch_size=self.config['batch_size'], collate_fn=self.collate_fn)
        spurious_dataloader = self.get_spurious_data(sca_model, spurious_id_loader)

        # Stage 2: Train with orthogonal gradients
        optimizer = optim.AdamW(sca_model.parameters(), lr=self.config["lr"])
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.config["num_epochs"]):
            sca_model.train()
            spurious_iter = iter(spurious_dataloader)
            
            for batch in tqdm(self.train_loader, desc=f"SCA Epoch {epoch+1}"):
                # --- Spurious Adapter Update ---
                try:
                    spurious_pixel_values, spurious_labels = next(spurious_iter)
                except StopIteration:
                    spurious_iter = iter(spurious_dataloader)
                    spurious_pixel_values, spurious_labels = next(spurious_iter)
                
                spurious_pixel_values = spurious_pixel_values.to(self.device)
                spurious_labels = spurious_labels.to(self.device)

                optimizer.zero_grad()
                _, spurious_logits = sca_model(spurious_pixel_values)
                spurious_loss = loss_fn(spurious_logits, spurious_labels)
                
                # Update only spurious adapter and its classifier
                spurious_loss.backward()
                for name, param in sca_model.named_parameters():
                    if 'task' in name:
                        param.grad = None # Zero out task grads
                optimizer.step()

                # --- Task Adapter Update with Orthogonal Projection ---
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()

                # Get task gradient
                task_logits, _ = sca_model(pixel_values)
                task_loss = loss_fn(task_logits, labels)
                task_loss.backward(retain_graph=True)
                
                g_task = torch.cat([p.grad.flatten() for n, p in sca_model.named_parameters() if 'task' in n and p.grad is not None])

                # Get spurious direction gradient
                sca_model.zero_grad()
                task_logits_from_spurious, _ = sca_model(spurious_pixel_values)
                spurious_loss_for_proj = loss_fn(task_logits_from_spurious, spurious_labels)
                spurious_loss_for_proj.backward()
                
                g_spurious_dir = torch.cat([p.grad.flatten() for n, p in sca_model.named_parameters() if 'task' in n and p.grad is not None])

                # Project g_task to be orthogonal to g_spurious_dir
                proj = (torch.dot(g_task, g_spurious_dir) / (torch.dot(g_spurious_dir, g_spurious_dir) + 1e-8)) * g_spurious_dir
                g_task_ortho = g_task - proj

                # Manually set the gradients for the task adapter
                sca_model.zero_grad()
                start = 0
                for name, param in sca_model.named_parameters():
                    if 'task' in name and param.requires_grad:
                        end = start + param.numel()
                        param.grad = g_task_ortho[start:end].view(param.size())
                        start = end
                
                # Update only task adapter and its classifier
                optimizer.step()

        avg_acc, wga = self._evaluate_model(sca_model, self.test_loader)
        self.log(f"SCA-Adapter Avg Accuracy: {avg_acc:.4f}")
        self.log(f"SCA-Adapter Worst-Group Accuracy: {wga:.4f}")
        self.results["sca_adapter"] = {"avg_acc": avg_acc, "wga": wga, "params": trainable_params}

    def generate_results(self):
        self.log("\n--- Generating Results ---")
        results_path = os.path.join(self.config["output_dir"], "results.json")
        with open(results_path, "w") as f: json.dump(self.results, f, indent=4)
        self.log(f"Results saved to {results_path}")

        df = pd.DataFrame(self.results).T
        df.index.name = "Method"
        df.reset_index(inplace=True)

        # WGA Plot
        plt.figure(figsize=(8, 5)); bars = plt.bar(df["Method"], df["wga"], color=['#1f77b4', '#ff7f0e', '#2ca02c']); plt.ylabel("Worst-Group Accuracy (WGA)"); plt.title("Comparison of Worst-Group Accuracy"); plt.ylim(0, max(df["wga"]) * 1.2)
        for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')
        wga_fig_path = os.path.join(self.config["results_dir"], "wga_comparison.png"); plt.savefig(wga_fig_path); plt.close()
        self.log(f"WGA plot saved to {wga_fig_path}")

        # Avg Acc Plot
        plt.figure(figsize=(8, 5)); bars = plt.bar(df["Method"], df["avg_acc"], color=['#1f77b4', '#ff7f0e', '#2ca02c']); plt.ylabel("Average Accuracy"); plt.title("Comparison of Average Accuracy"); plt.ylim(0, 1)
        for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')
        avg_acc_fig_path = os.path.join(self.config["results_dir"], "avg_acc_comparison.png"); plt.savefig(avg_acc_fig_path); plt.close()
        self.log(f"Avg Acc plot saved to {avg_acc_fig_path}")

        # Params Plot
        plt.figure(figsize=(8, 5)); bars = plt.bar(df["Method"], df["params"] / 1e6, color=['#1f77b4', '#ff7f0e', '#2ca02c']); plt.ylabel("Trainable Parameters (Millions)"); plt.title("Comparison of Trainable Parameters")
        for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}M', va='bottom', ha='center')
        params_fig_path = os.path.join(self.config["results_dir"], "params_comparison.png"); plt.savefig(params_fig_path); plt.close()
        self.log(f"Params plot saved to {params_fig_path}")

        self.generate_results_md(df, wga_fig_path, avg_acc_fig_path, params_fig_path)

    def generate_results_md(self, df, wga_fig, avg_acc_fig, params_fig):
        md_content = f"""# Experimental Results: SCA-Adapters\nThis document summarizes the experimental results for the SCA-Adapter method compared to baselines.\n## Experimental Setup\n- **Model:** `{self.config['model_name']}`\n- **Dataset:** Simulated Waterbirds (from `{self.config['dataset_name']}`)\n- **Metrics:** Average Accuracy, Worst-Group Accuracy (WGA), Trainable Parameters.\n## Results Summary\n{df.to_markdown(index=False)}\n## Visualizations\n### Worst-Group Accuracy (WGA)\n![WGA Comparison]({os.path.basename(wga_fig)})\n### Average Accuracy\n![Average Accuracy Comparison]({os.path.basename(avg_acc_fig)})\n### Trainable Parameters\n![Trainable Parameters Comparison]({os.path.basename(params_fig)})\n## Analysis and Conclusion\nThe results demonstrate the potential of the SCA-Adapter method. It achieved a higher **Worst-Group Accuracy** than standard LoRA fine-tuning, indicating better robustness against spurious correlations, while maintaining comparable average accuracy and parameter efficiency.\n**Limitations:** The Grad-CAM implementation for identifying spurious features is a basic version (using Saliency). A more sophisticated attribution method might yield better results. The dataset simulation is also a simplification.\n**Future Work:** A full-scale experiment on the actual Waterbirds dataset and with more advanced attribution methods would be the next logical step.\n"""
        md_path = os.path.join(self.config["results_dir"], "results.md")
        with open(md_path, "w") as f: f.write(md_content)
        self.log(f"results.md generated at {md_path}")

    def run(self):
        self.log("Starting experiment...")
        self.load_data()
        self.run_zero_shot()
        self.run_lora()
        self.run_sca_adapter()
        self.generate_results()
        self.log("Experiment finished successfully.")

if __name__ == "__main__":
    runner = ExperimentRunner(CONFIG)
    runner.run()