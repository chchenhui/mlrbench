
import torch
from sentence_transformers import SentenceTransformer, models

class BiAlignModel:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.device)

    def get_model(self):
        return self.model

    def save(self, path):
        self.model.save(path)

def get_base_multilingual_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

def get_distiluse_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1', device=device)

def get_embeddings(model, sentences, batch_size=32):
    if isinstance(model, SentenceTransformer):
        return model.encode(sentences, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    else:
        raise TypeError(f"Unknown model type: {type(model)}")

if __name__ == '__main__':
    print("Initializing Bi-Align model...")
    bialign = BiAlignModel()
    print("Bi-Align model initialized.")

    print("Initializing Base Multilingual model...")
    base_multi = get_base_multilingual_model()
    print("Base Multilingual model initialized.")

    print("Initializing DistilUSE model...")
    distiluse = get_distiluse_model()
    print("DistilUSE model initialized.")

    test_sentences = ["This is a test sentence.", "Ceci est une phrase de test."]
    
    bialign_embeddings = get_embeddings(bialign.get_model(), test_sentences)
    print(f"Bi-Align embeddings shape: {bialign_embeddings.shape}")

    base_multi_embeddings = get_embeddings(base_multi, test_sentences)
    print(f"Base Multilingual embeddings shape: {base_multi_embeddings.shape}")

    distiluse_embeddings = get_embeddings(distiluse, test_sentences)
    print(f"DistilUSE embeddings shape: {distiluse_embeddings.shape}")
