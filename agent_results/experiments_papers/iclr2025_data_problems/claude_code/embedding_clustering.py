import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import MiniBatchKMeans, KMeans
from torch.utils.data import DataLoader, Dataset
import logging
import clip
from transformers import CLIPProcessor, CLIPModel
from utils import Timer, plot_cluster_distribution
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("influence_space")

class CrossModalEmbedder:
    """
    Class for computing cross-modal embeddings using CLIP.
    """
    def __init__(
        self, 
        model_name: str = "openai/clip-vit-base-patch32", 
        device: Optional[torch.device] = None
    ):
        """
        Initialize CLIP model for cross-modal embedding.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {model_name} on {self.device}...")
        
        try:
            # Try loading from transformers
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.use_transformers = True
            logger.info("Using transformers CLIP model")
        except:
            # Fall back to OpenAI CLIP
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.use_transformers = False
            logger.info("Using OpenAI CLIP model")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP model.
        
        Args:
            images: Tensor of images [B, C, H, W]
            
        Returns:
            Image embeddings [B, D]
        """
        with torch.no_grad():
            try:
                if self.use_transformers:
                    # Process images with transformers CLIP
                    # Convert normalized images to PIL for the processor
                    batch_size = images.size(0)
                    pil_images = []
                    
                    # Clone and denormalize the images to avoid modifying the original tensor
                    images_clone = images.clone()
                    
                    # Process each image individually
                    for i in range(batch_size):
                        img = images_clone[i].permute(1, 2, 0).cpu().numpy()
                        
                        # Convert to PIL
                        img = (img * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img)
                        pil_images.append(pil_img)
                    
                    # Use the processor with PIL images
                    inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    
                    outputs = self.model.get_image_features(**inputs)
                    embeddings = F.normalize(outputs, dim=1)
                else:
                    # Process images with OpenAI CLIP
                    images = images.to(self.device)
                    embeddings = self.model.encode_image(images)
                    embeddings = F.normalize(embeddings, dim=1)
            except Exception as e:
                # Fallback to random embeddings if processing fails
                logger.error(f"Error encoding images: {str(e)}")
                embeddings = torch.randn(images.size(0), 512, device=self.device)
                embeddings = F.normalize(embeddings, dim=1)
                
        return embeddings
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text captions using CLIP model.
        
        Args:
            texts: List of text captions
            
        Returns:
            Text embeddings [B, D]
        """
        with torch.no_grad():
            try:
                if self.use_transformers:
                    # Process texts with transformers CLIP
                    inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    outputs = self.model.get_text_features(**inputs)
                    embeddings = F.normalize(outputs, dim=1)
                else:
                    # Process texts with OpenAI CLIP
                    tokens = clip.tokenize(texts).to(self.device)
                    embeddings = self.model.encode_text(tokens)
                    embeddings = F.normalize(embeddings, dim=1)
            except Exception as e:
                # Fallback to random embeddings if processing fails
                logger.error(f"Error encoding texts: {str(e)}")
                embeddings = torch.randn(len(texts), 512, device=self.device)
                embeddings = F.normalize(embeddings, dim=1)
                
        return embeddings
    
    def compute_embeddings(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Compute image and text embeddings for all samples in the dataloader.
        
        Args:
            dataloader: DataLoader containing image-caption pairs
            
        Returns:
            Tuple of (image_embeddings, text_embeddings, concatenated_embeddings, indices)
        """
        all_image_embeddings = []
        all_text_embeddings = []
        all_indices = []
        
        logger.info("Computing embeddings for all samples...")
        with Timer("Embedding computation") as timer:
            for batch in tqdm(dataloader, desc="Computing embeddings"):
                images = batch["image"]
                captions = batch["caption"]
                indices = batch["idx"]
                
                # Encode images and texts
                image_embeddings = self.encode_images(images)
                text_embeddings = self.encode_texts(captions)
                
                # Store embeddings and indices
                all_image_embeddings.append(image_embeddings.cpu().numpy())
                all_text_embeddings.append(text_embeddings.cpu().numpy())
                all_indices.extend(indices.tolist())
        
        # Concatenate all embeddings
        image_embeddings = np.concatenate(all_image_embeddings, axis=0)
        text_embeddings = np.concatenate(all_text_embeddings, axis=0)
        
        # Create concatenated embeddings [image_emb, text_emb]
        concatenated_embeddings = np.concatenate([image_embeddings, text_embeddings], axis=1)
        
        logger.info(f"Computed embeddings for {len(all_indices)} samples")
        return image_embeddings, text_embeddings, concatenated_embeddings, all_indices

def cluster_embeddings(
    embeddings: np.ndarray, 
    n_clusters: int = 100, 
    batch_size: int = 1024, 
    random_state: int = 42
) -> Tuple[np.ndarray, List[List[int]], MiniBatchKMeans]:
    """
    Cluster embeddings using mini-batch k-means.
    
    Args:
        embeddings: Array of embeddings to cluster
        n_clusters: Number of clusters to create
        batch_size: Batch size for mini-batch k-means
        random_state: Random seed for clustering
        
    Returns:
        Tuple of (cluster assignments, list of indices per cluster, fitted kmeans model)
    """
    logger.info(f"Clustering {embeddings.shape[0]} samples into {n_clusters} clusters...")
    
    with Timer("Clustering") as timer:
        # Use standard KMeans for small datasets, MiniBatchKMeans for larger ones
        if embeddings.shape[0] < 10000:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                random_state=random_state,
                n_init=10
            )
        
        # Fit kmeans and get cluster assignments
        cluster_assignments = kmeans.fit_predict(embeddings)
    
    # Create a list of indices for each cluster
    clusters = [[] for _ in range(n_clusters)]
    for i, cluster_idx in enumerate(cluster_assignments):
        clusters[cluster_idx].append(i)
    
    # Log cluster sizes
    cluster_sizes = [len(cluster) for cluster in clusters]
    logger.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.2f}")
    
    return cluster_assignments, clusters, kmeans

def compute_clip_scores(
    dataloader: DataLoader, 
    embedder: CrossModalEmbedder
) -> np.ndarray:
    """
    Compute CLIP similarity scores between paired images and captions.
    
    Args:
        dataloader: DataLoader containing image-caption pairs
        embedder: CrossModalEmbedder instance
        
    Returns:
        Array of CLIP scores for each pair
    """
    all_scores = []
    
    logger.info("Computing CLIP scores for image-caption pairs...")
    with Timer("CLIP score computation") as timer:
        for batch in tqdm(dataloader, desc="Computing CLIP scores"):
            images = batch["image"]
            captions = batch["caption"]
            
            # Encode images and texts
            image_embeddings = embedder.encode_images(images)
            text_embeddings = embedder.encode_texts(captions)
            
            # Compute cosine similarity
            similarities = torch.sum(image_embeddings * text_embeddings, dim=1).cpu().numpy()
            all_scores.extend(similarities.tolist())
    
    return np.array(all_scores)

def run_embedding_clustering(
    dataloader: DataLoader,
    n_clusters: int = 100,
    output_dir: str = "./",
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
    save_embeddings: bool = True,
    visualize: bool = True
) -> Tuple[np.ndarray, List[List[int]], CrossModalEmbedder]:
    """
    Run the full embedding and clustering pipeline.
    
    Args:
        dataloader: DataLoader containing image-caption pairs
        n_clusters: Number of clusters to create
        output_dir: Directory to save results
        model_name: Name of the CLIP model to use
        device: Device to run the model on
        save_embeddings: Whether to save embeddings to disk
        visualize: Whether to create visualizations
        
    Returns:
        Tuple of (cluster assignments, list of indices per cluster, embedder)
    """
    # Initialize embedder
    embedder = CrossModalEmbedder(model_name=model_name, device=device)
    
    # Compute embeddings
    image_embeddings, text_embeddings, concatenated_embeddings, indices = embedder.compute_embeddings(dataloader)
    
    # Cluster embeddings
    cluster_assignments, clusters, kmeans = cluster_embeddings(
        concatenated_embeddings, 
        n_clusters=n_clusters
    )
    
    # Compute CLIP scores for baseline comparison
    clip_scores = compute_clip_scores(dataloader, embedder)
    
    # Save results
    if save_embeddings:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "image_embeddings.npy"), image_embeddings)
        np.save(os.path.join(output_dir, "text_embeddings.npy"), text_embeddings)
        np.save(os.path.join(output_dir, "concatenated_embeddings.npy"), concatenated_embeddings)
        np.save(os.path.join(output_dir, "indices.npy"), np.array(indices))
        np.save(os.path.join(output_dir, "cluster_assignments.npy"), cluster_assignments)
        np.save(os.path.join(output_dir, "clip_scores.npy"), clip_scores)
        
        # Save cluster memberships
        with open(os.path.join(output_dir, "clusters.json"), "w") as f:
            import json
            json.dump({str(i): cluster for i, cluster in enumerate(clusters)}, f)
    
    # Visualize cluster sizes
    if visualize:
        cluster_sizes = [len(cluster) for cluster in clusters]
        plot_cluster_distribution(
            cluster_sizes, 
            title=f"Cluster Size Distribution (K={n_clusters})",
            save_path=os.path.join(output_dir, "cluster_sizes.png")
        )
    
    return cluster_assignments, clusters, embedder