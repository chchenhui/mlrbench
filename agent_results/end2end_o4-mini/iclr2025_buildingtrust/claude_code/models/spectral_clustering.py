"""
Implementation of the hierarchical spectral clustering component for the 
Cluster-Driven Certified Unlearning method.
"""

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from scipy import sparse
from scipy.sparse.linalg import eigsh


class RepresentationClustering:
    """
    Class for clustering the model's knowledge by segmenting hidden-layer activations
    into semantically coherent clusters.
    """
    
    def __init__(self, n_clusters=10, embedding_dim=64, sigma=1.0):
        """
        Initialize the clustering module.
        
        Args:
            n_clusters (int): Number of clusters to form
            embedding_dim (int): Dimension of the spectral embedding
            sigma (float): Bandwidth parameter for the similarity matrix
        """
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.sigma = sigma
        self.clusters = None
        self.cluster_centers = None
        self.spectral_embedding = None
        
    def compute_similarity_matrix(self, activations):
        """
        Compute pairwise affinities between hidden-layer activations.
        
        Args:
            activations (torch.Tensor): Hidden-layer activations [n_samples, dimension]
            
        Returns:
            S (sparse.csr_matrix): Sparse similarity matrix
        """
        n_samples = activations.shape[0]
        
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
            
        # We'll use a sparse approach with k-nearest neighbors to avoid memory issues
        from sklearn.neighbors import NearestNeighbors
        
        # Use 50 neighbors or fewer if we have fewer samples
        k = min(50, n_samples - 1)
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(activations)
        distances, indices = nbrs.kneighbors(activations)
        
        # Convert distances to similarities
        similarities = np.exp(-distances**2 / (2 * self.sigma**2))
        
        # Create sparse similarity matrix
        rows = np.repeat(np.arange(n_samples), k)
        cols = indices.flatten()
        data = similarities.flatten()
        
        S = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        
        # Make the matrix symmetric
        S = (S + S.T) / 2
        
        return S
        
    def compute_graph_laplacian(self, S):
        """
        Form the normalized graph Laplacian matrix.
        
        Args:
            S (sparse.csr_matrix): Similarity matrix
            
        Returns:
            L (sparse.csr_matrix): Graph Laplacian
            D (sparse.csr_matrix): Degree matrix
        """
        # Compute degree matrix
        degrees = np.array(S.sum(axis=1)).flatten()
        D = sparse.diags(degrees)
        
        # Compute Laplacian
        L = D - S
        
        return L, D
        
    def compute_spectral_embedding(self, L, D):
        """
        Compute the spectral embedding by solving the generalized eigenvalue problem.
        
        Args:
            L (sparse.csr_matrix): Graph Laplacian
            D (sparse.csr_matrix): Degree matrix
            
        Returns:
            U (np.ndarray): Spectral embedding
        """
        # Solve generalized eigenvalue problem L u = λ D u
        # We can transform this to a standard eigenvalue problem: D^(-1/2) L D^(-1/2) v = λ v
        # where v = D^(1/2) u
        
        # Compute D^(-1/2)
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.array(D.sum(axis=1)).flatten()))
        
        # Compute normalized Laplacian
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt
        
        # Compute eigenvectors of normalized Laplacian
        # We want the smallest eigenvalues (which correspond to the smoothest eigenvectors)
        eigenvalues, eigenvectors = eigsh(L_norm, k=self.embedding_dim, which='SM')
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normalize rows to unit norm
        row_norms = np.sqrt(np.sum(eigenvectors ** 2, axis=1))
        U = eigenvectors / row_norms[:, np.newaxis]
        
        return U
        
    def cluster_embeddings(self, U):
        """
        Apply hierarchical agglomerative clustering on spectral embeddings.
        
        Args:
            U (np.ndarray): Spectral embedding
            
        Returns:
            cluster_assignments (np.ndarray): Cluster assignments for each sample
        """
        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'  # Ward's method minimizes variance within clusters
        )
        
        cluster_assignments = clustering.fit_predict(U)
        
        return cluster_assignments
        
    def fit(self, activations):
        """
        Perform the full clustering process.
        
        Args:
            activations (torch.Tensor or np.ndarray): Hidden-layer activations [n_samples, dimension]
            
        Returns:
            self: The fitted clustering object
        """
        # Compute similarity matrix
        S = self.compute_similarity_matrix(activations)
        
        # Compute graph Laplacian
        L, D = self.compute_graph_laplacian(S)
        
        # Compute spectral embedding
        U = self.compute_spectral_embedding(L, D)
        self.spectral_embedding = U
        
        # Cluster the embeddings
        self.clusters = self.cluster_embeddings(U)
        
        # Compute cluster centers (using original activations)
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
            
        self.cluster_centers = np.zeros((self.n_clusters, activations.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(self.clusters == k) > 0:
                self.cluster_centers[k] = activations[self.clusters == k].mean(axis=0)
                
        return self
        
    def get_cluster_basis(self, activations, cluster_idx):
        """
        Get the orthonormal basis for a specific cluster.
        
        Args:
            activations (torch.Tensor or np.ndarray): Hidden-layer activations
            cluster_idx (int): Cluster index
            
        Returns:
            U_k (np.ndarray): Orthonormal basis for the cluster
        """
        if self.clusters is None:
            raise ValueError("Clustering has not been performed yet. Call fit() first.")

        # Convert activations to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
            
        # Get samples belonging to the cluster
        cluster_mask = self.clusters == cluster_idx
        cluster_activations = activations[cluster_mask]
        
        if cluster_activations.shape[0] == 0:
            raise ValueError(f"No samples in cluster {cluster_idx}")
            
        # Compute SVD to get orthonormal basis
        U, _, _ = np.linalg.svd(cluster_activations - cluster_activations.mean(axis=0), full_matrices=False)
        
        # Return orthonormal basis
        return U