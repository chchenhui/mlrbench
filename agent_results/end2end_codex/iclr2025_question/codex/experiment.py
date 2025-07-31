import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

class LossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
            self.eval_losses = []
        if 'loss' in logs:
            self.train_losses.append((state.epoch, logs['loss']))
        if 'eval_loss' in logs:
            self.eval_losses.append((state.epoch, logs['eval_loss']))

from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--method', type=str, choices=['baseline', 'head_only'], default='baseline')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    ds = load_dataset('glue', 'sst2')
    # Use small subset for speed
    train_ds = ds['train'].shuffle(seed=42).select(range(200))
    eval_ds = ds['validation'].shuffle(seed=42).select(range(100))

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)
    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)
    for split in [train_ds, eval_ds]:
        split.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    if args.method == 'head_only':
        # freeze all except classification head
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='no',
        disable_tqdm=True,
        load_best_model_at_end=False,
        report_to=[]
    )

    loss_callback = LossCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )

    # Train
    train_result = trainer.train()
    trainer.save_state()

    # Evaluate
    eval_result = trainer.evaluate()

    # Save metrics
    results = {
        'train_loss': loss_callback.train_losses,
        'eval_loss': loss_callback.eval_losses,
        'final_eval': eval_result
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
