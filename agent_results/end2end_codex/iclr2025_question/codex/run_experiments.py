import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    methods = ['baseline', 'head_only']
    results_dirs = []
    log_path = os.path.join(base_dir, 'log.txt')
    with open(log_path, 'w') as log_file:
        for method in methods:
            out_dir = os.path.join(base_dir, 'results', method)
            os.makedirs(out_dir, exist_ok=True)
            cmd = [
                sys.executable, os.path.join(base_dir, 'experiment.py'),
                '--method', method,
                '--output_dir', out_dir
            ]
            log_file.write(f"Running {method}...\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log_file.write(line)
            proc.wait()
            if proc.returncode != 0:
                log_file.write(f"Error: {method} exited with {proc.returncode}\n")
                raise RuntimeError(f"Experiment {method} failed")
            results_dirs.append(out_dir)
        # Plot results
        log_file.write("Plotting results...\n")
        plot_cmd = [
            sys.executable, os.path.join(base_dir, 'plot_results.py'),
            '--input_dirs', *results_dirs,
            '--output_dir', os.path.join(base_dir, 'results')
        ]
        proc = subprocess.Popen(plot_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log_file.write(line)
        proc.wait()
        if proc.returncode != 0:
            log_file.write(f"Plotting failed with {proc.returncode}\n")
            raise RuntimeError("Plotting failed")
    # Gather results to root/results
    import shutil
    root_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
    final_dir = os.path.join(root_dir, 'results')
    os.makedirs(final_dir, exist_ok=True)
    # Copy log
    shutil.copy(os.path.join(base_dir, 'log.txt'), os.path.join(final_dir, 'log.txt'))
    # Copy generated files
    src_res = os.path.join(base_dir, 'results')
    for fname in os.listdir(src_res):
        full = os.path.join(src_res, fname)
        if os.path.isfile(full):
            shutil.copy(full, os.path.join(final_dir, fname))
    # Generate results.md
    md_path = os.path.join(final_dir, 'results.md')
    with open(md_path, 'w') as md:
        md.write('# Experiment Results Summary\n\n')
        md.write('## Experimental Setup\n')
        md.write('- Dataset: SST-2 subset (200 train, 100 validation)\n')
        md.write('- Model: distilbert-base-uncased\n')
        md.write('- Methods: full fine-tuning (baseline), head-only fine-tuning (head_only)\n\n')
        # Table from CSV
        csv_path = os.path.join(src_res, 'results.csv')
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            md.write('## Results Table\n')
            md.write(df.to_markdown(index=False))
            md.write('\n\n')
        # Figures
        md.write('## Figures\n')
        md.write('### Loss Curves\n')
        md.write('![](loss_curve.png)\n\n')
        md.write('### Final Evaluation Metrics\n')
        md.write('![](metrics.png)\n\n')
        md.write('## Discussion\n')
        md.write('The results show the performance of the two methods. Please refer to the figures and table above.\n')
        md.write('The head-only fine-tuning method is expected to train faster but may achieve lower accuracy compared to full fine-tuning.\n')
        md.write('### Limitations and Future Work\n')
        md.write('- Limited dataset size and epochs.\n')
        md.write('- Future work: larger datasets, more epochs, additional baselines.\n')

if __name__ == '__main__':
    main()
