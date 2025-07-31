#!/usr/bin/env python
"""
Automate running ERM and AIFS experiments and generate plots.
"""
import os
import subprocess
import logging

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')  # not used, uses default in train.py
    results_dir = os.path.join(base_dir, 'results_raw')
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(base_dir, 'log.txt')
    log_f = open(log_path, 'w')
    methods = ['ERM', 'AIFS']
    for method in methods:
        out_dir = os.path.join(results_dir, method)
        os.makedirs(out_dir, exist_ok=True)
        cmd = ['python', 'train.py', '--method', method,
               '--output_dir', out_dir]
        logging.info(f"Running {method}...")
        proc = subprocess.Popen(cmd, cwd=base_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            decoded = line.decode()
            log_f.write(decoded)
            log_f.flush()
        proc.wait()
    log_f.close()
    # generate plots
    from plot import plot_results
    fig_dir = os.path.join(base_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plot_results(results_dir, fig_dir)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run()
