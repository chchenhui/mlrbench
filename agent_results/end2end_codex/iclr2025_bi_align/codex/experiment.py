#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
import matplotlib.pyplot as plt

def simulate_user(theta, phi_i, phi_j):
    """Simulate user feedback y=1 if prefers i over j, else 0."""
    diff = theta.dot(phi_i - phi_j)
    p = 1 / (1 + np.exp(-diff))
    return 1 if random.random() < p else 0

class BayesianPreferenceModel:
    def __init__(self, dim):
        self.dim = dim
        self.mu = np.zeros(dim)
        self.Sigma = np.eye(dim)
        self.Sigma_inv = np.eye(dim)
    def predict(self, phi):
        mu = self.mu.dot(phi)
        var = phi.dot(self.Sigma).dot(phi)
        return mu, var
    def update(self, phi_diff, y):
        # approximate Bayesian update with Laplace approximation
        p = 1 / (1 + np.exp(-self.mu.dot(phi_diff)))
        W = p * (1 - p)
        self.Sigma_inv = self.Sigma_inv + W * np.outer(phi_diff, phi_diff)
        self.Sigma = np.linalg.inv(self.Sigma_inv)
        grad = (y - p) * phi_diff
        self.mu = self.mu + self.Sigma.dot(grad)

def run_experiment(num_items=50, dim=5, K=10, T=100, threshold=2.0):
    phi = np.random.randn(num_items, dim)
    theta_true = np.random.randn(dim)
    results = {'static': [], 'passive': [], 'udca': []}
    queries = {'static': 0, 'passive': 0, 'udca': 0}
    models = {
        'static': BayesianPreferenceModel(dim),
        'passive': BayesianPreferenceModel(dim),
        'udca': BayesianPreferenceModel(dim),
    }
    tau = threshold
    for t in range(T):
        idx = np.random.choice(num_items, K, replace=False)
        phis = phi[idx]
        for name, model in models.items():
            mus = np.array([model.predict(p)[0] for p in phis])
            a_idx = np.argmax(mus)
            a = phis[a_idx]
            util = theta_true.dot(a)
            results[name].append(util)
            if name == 'static':
                continue
            if name == 'passive':
                i, j = random.sample(range(K), 2)
                y = simulate_user(theta_true, phis[i], phis[j])
                models[name].update(phis[i] - phis[j], y)
                queries[name] += 1
            if name == 'udca':
                vars_ = np.array([model.predict(p)[1] for p in phis])
                if vars_.max() > tau:
                    p_i = np.argmax(vars_)
                    p_j = random.choice([x for x in range(K) if x != p_i])
                    y = simulate_user(theta_true, phis[p_i], phis[p_j])
                    models[name].update(phis[p_i] - phis[p_j], y)
                    queries[name] += 1
                tau *= 0.99
    out_dir = os.path.join(os.getcwd(), 'codex')
    os.makedirs(out_dir, exist_ok=True)
    json.dump({'results': results, 'queries': queries}, open(os.path.join(out_dir, 'results.json'), 'w'))
    plt.figure()
    T_range = np.arange(T)
    for name, vals in results.items():
        plt.plot(np.cumsum(vals) / (T_range + 1), label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Avg true utility')
    plt.title('Decision Quality over Time')
    plt.legend()
    fig1 = os.path.join(out_dir, 'decision_quality.png')
    plt.savefig(fig1)
    plt.close()
    plt.figure()
    pces = {}
    for name, model in models.items():
        mus = phi.dot(model.mu)
        pces[name] = float(np.mean((phi.dot(theta_true) - mus)**2))
    names = list(pces.keys())
    vals = [pces[n] for n in names]
    plt.bar(names, vals)
    plt.ylabel('Preference Calibration Error')
    plt.title('Final PCE')
    fig2 = os.path.join(out_dir, 'pce.png')
    plt.savefig(fig2)
    plt.close()
    return out_dir

if __name__ == '__main__':
    import logging
    log_path = os.path.join('codex', 'log.txt')
    os.makedirs('codex', exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Starting experiment')
    out = run_experiment()
    logging.info(f'Results saved to {out}')
    print('Experiment completed.')
