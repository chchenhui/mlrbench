import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    # load data
    all_losses = {}
    for dir_path in args.input_dirs:
        name = os.path.basename(dir_path)
        with open(os.path.join(dir_path, 'results.json')) as f:
            data = json.load(f)
        # losses
        train = data['train_loss']
        evals = data['eval_loss']
        all_losses[name] = {'train': train, 'eval': evals}
        # final metrics
        final = data['final_eval']
        records.append({
            'method': name,
            'accuracy': final.get('eval_accuracy', None),
            'f1': final.get('eval_f1', None)
        })

    # Plot loss curves
    plt.figure()
    for method, losses in all_losses.items():
        train_losses = losses['train']
        eval_losses = losses['eval']
        # align lengths: ignore extra eval losses
        n = min(len(train_losses), len(eval_losses))
        epochs, train_vals = zip(*train_losses[:n])
        _, eval_vals = zip(*eval_losses[:n])
        plt.plot(epochs, train_vals, marker='o', label=f'{method} train')
        plt.plot(epochs, eval_vals, marker='x', label=f'{method} eval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    loss_fig = os.path.join(args.output_dir, 'loss_curve.png')
    plt.savefig(loss_fig)
    plt.close()

    # Plot metrics
    df = pd.DataFrame(records)
    metrics_fig = os.path.join(args.output_dir, 'metrics.png')
    ax = df.set_index('method').plot.bar(rot=0)
    ax.set_ylabel('Score')
    ax.set_title('Final Evaluation Metrics')
    plt.tight_layout()
    plt.savefig(metrics_fig)
    plt.close()

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f'Plots saved to {loss_fig} and {metrics_fig}')
    print(f'Results table saved to {csv_path}')

if __name__ == '__main__':
    main()
