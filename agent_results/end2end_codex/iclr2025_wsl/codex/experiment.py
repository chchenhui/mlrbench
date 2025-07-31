import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Setup logging
os.makedirs('codex', exist_ok=True)
logging.basicConfig(filename='codex/log.txt', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_tasks(num_tasks=10, n_per=100):
    tasks = []
    for _ in range(num_tasks):
        mu0 = np.random.uniform(-5, 5, size=2)
        mu1 = np.random.uniform(-5, 5, size=2)
        x0 = np.random.randn(n_per, 2) + mu0
        x1 = np.random.randn(n_per, 2) + mu1
        X = np.vstack([x0, x1]).astype(np.float32)
        y = np.hstack([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)
        desc = np.concatenate([mu0, mu1]).astype(np.float32)
        tasks.append((X, y, desc))
    return tasks

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

def train_classifier(X, y, epochs=20):
    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model

def extract_weights(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy()

class DiffusionModel(nn.Module):
    def __init__(self, w_dim, desc_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim+desc_dim+1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, w_dim)
        )
    def forward(self, w_noisy, t, desc):
        # w_noisy: [B, w_dim], t: [B,1], desc: [B, desc_dim]
        inp = torch.cat([w_noisy, t, desc], dim=1)
        return self.net(inp)

def train_diffusion(weights, descs, timesteps=50, epochs=200):
    # weights: [N, w_dim], descs: [N, desc_dim]
    w_dim = weights.shape[1]
    desc_dim = descs.shape[1]
    model = DiffusionModel(w_dim, desc_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    beta = np.linspace(1e-4, 0.02, timesteps)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    # convert schedules to torch tensors
    alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    alpha_bar = torch.tensor(alpha_bar, device=device, dtype=torch.float32)
    Ws = torch.from_numpy(weights).to(device)
    Ds = torch.from_numpy(descs).to(device)
    for ep in range(epochs):
        idx = np.random.permutation(len(Ws))
        for i in idx:
            w = Ws[i:i+1]
            d = Ds[i:i+1]
            t = np.random.randint(0, timesteps)
            a_bar = alpha_bar[t]
            noise = torch.randn_like(w)
            w_noisy = torch.sqrt(a_bar)*w + torch.sqrt(1-a_bar)*noise
            t_tensor = torch.tensor([[t/timesteps]], device=device, dtype=torch.float32)
            noise_pred = model(w_noisy, t_tensor, d)
            loss = ((noise_pred - noise)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            logger.info(f'Diffusion epoch {ep} loss {loss.item():.4f}')
    return model, alpha, alpha_bar

def sample_diffusion(model, alpha, alpha_bar, desc, timesteps=50):
    model.eval()
    w_dim = model.net[-1].out_features
    x = torch.randn(1, w_dim, device=device)
    d = torch.from_numpy(desc.reshape(1,-1)).to(device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([[t / timesteps]], device=device, dtype=torch.float32)
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = 1 - alpha_t
        noise_pred = model(x, t_tensor, d)
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
    return x.detach().cpu().numpy().reshape(-1)

def evaluate(model_zoo, diffusion_model, alpha, alpha_bar):
    # single new task
    X, y, desc = generate_tasks(1, n_per=100)[0]
    # baseline
    bl_model = MLP().to(device)
    bl_opt = optim.Adam(bl_model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    bl_losses = []
    # init from diffusion
    w_init = sample_diffusion(diffusion_model, alpha, alpha_bar, desc)
    dm = MLP().to(device)
    # load weights
    with torch.no_grad():
        vec = torch.from_numpy(w_init).to(device)
        torch.nn.utils.vector_to_parameters(vec, dm.parameters())
    dm_opt = optim.Adam(dm.parameters(), lr=0.01)
    dm_losses = []
    epochs = 20
    for ep in range(epochs):
        bl_loss_sum = 0; dm_loss_sum = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
            # baseline step
            bl_opt.zero_grad()
            lb = loss_fn(bl_model(xb), yb); lb.backward(); bl_opt.step()
            bl_loss_sum += lb.item()
            # diff init step
            dm_opt.zero_grad()
            ld = loss_fn(dm(xb), yb); ld.backward(); dm_opt.step()
            dm_loss_sum += ld.item()
        bl_losses.append(bl_loss_sum/len(dl))
        dm_losses.append(dm_loss_sum/len(dl))
    # save curves
    np.save('codex/bl_losses.npy', bl_losses)
    np.save('codex/dm_losses.npy', dm_losses)
    # plot
    plt.figure()
    plt.plot(bl_losses, label='random init')
    plt.plot(dm_losses, label='diffusion init')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training Loss')
    plt.savefig('codex/loss_curve.png')
    return bl_losses, dm_losses

def main():
    logger.info('Generating tasks')
    tasks = generate_tasks(num_tasks=20, n_per=100)
    weights, descs = [], []
    logger.info('Training model zoo')
    for i, (X, y, desc) in enumerate(tasks):
        model = train_classifier(X, y, epochs=20)
        w = extract_weights(model)
        weights.append(w); descs.append(desc)
        logger.info(f'Task {i} done')
    weights = np.stack(weights)
    descs = np.stack(descs)
    logger.info('Training diffusion model')
    diffusion_model, alpha, alpha_bar = train_diffusion(weights, descs, timesteps=50, epochs=200)
    logger.info('Evaluating')
    bl, dm = evaluate(weights, diffusion_model, alpha, alpha_bar)
    # save results
    res = {'bl_losses': bl, 'dm_losses': dm}
    with open('codex/results.json', 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    main()
