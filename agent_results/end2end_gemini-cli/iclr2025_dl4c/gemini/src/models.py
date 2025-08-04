import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead

def get_base_model(model_name="gpt2"):
    """
    Loads a pre-trained base model and tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def get_reward_model(model_name="gpt2"):
    """
    Loads a pre-trained model for sequence classification to be used as a reward model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model, tokenizer

def get_ppo_model(model_name="gpt2"):
    """
    Loads a pre-trained model with a value head for PPO training.
    """
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

if __name__ == '__main__':
    base_model, base_tokenizer = get_base_model()
    print("Base Model Loaded:", base_model.config.model_type)
    
    reward_model, reward_tokenizer = get_reward_model()
    print("Reward Model Loaded:", reward_model.config.model_type)

    ppo_model, ppo_tokenizer = get_ppo_model()
    print("PPO Model Loaded:", ppo_model.config.model_type)