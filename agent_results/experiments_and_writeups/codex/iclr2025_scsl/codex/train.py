import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from data import ColoredMNIST
from model import CNNEncoder, Classifier

def train_erm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Prepare data
    dataset = ColoredMNIST(root=args.data_dir, train=True)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    # Model
    encoder = CNNEncoder(latent_dim=args.latent_dim).to(device)
    clf = Classifier(latent_dim=args.latent_dim, num_classes=2).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(clf.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Logs
    history = []
    for epoch in range(1, args.epochs+1):
        # train
        encoder.train(); clf.train()
        running_loss = 0; correct = 0; total = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = encoder(x)
            logits = clf(z)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        # val
        encoder.eval(); clf.eval()
        v_loss = 0; v_corr = 0; v_tot = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                z = encoder(x); logits = clf(z)
                loss = criterion(logits, y)
                v_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                v_corr += (preds == y).sum().item()
                v_tot += y.size(0)
        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        logging.info(f"ERM Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc})
    # save history
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    hist_path = os.path.join(args.output_dir, 'erm_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    # evaluate on test set (distribution shift)
    test_ds = ColoredMNIST(root=args.data_dir, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    encoder.eval(); clf.eval()
    t_loss = 0; t_corr = 0; t_tot = 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x); logits = clf(z)
            loss = criterion(logits, y)
            t_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            t_corr += (preds == y).sum().item()
            t_tot += y.size(0)
    test_loss = t_loss / t_tot
    test_acc = t_corr / t_tot
    logging.info(f"ERM Test: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    # save test results
    test_path = os.path.join(args.output_dir, 'erm_test.json')
    with open(test_path, 'w') as f:
        json.dump({'test_loss': test_loss, 'test_acc': test_acc}, f)

def train_aifs(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ColoredMNIST(root=args.data_dir, train=True)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    encoder = CNNEncoder(latent_dim=args.latent_dim).to(device)
    clf = Classifier(latent_dim=args.latent_dim, num_classes=2).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(clf.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    # initialize intervention dims
    D = args.latent_dim
    k = args.k_dims
    priority_dims = torch.randperm(D)[:k].tolist()
    # buffers for grad-based attribution
    grad_sums = torch.zeros(D)
    grad_counts = 0
    history = []
    for epoch in range(1, args.epochs+1):
        encoder.train(); clf.train()
        running_loss = running_corr = running_tot = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            z = encoder(x)
            logits = clf(z)
            loss_cls = criterion(logits, y)
            # gradient-based attribution on z
            z2 = z.detach().requires_grad_(True)
            logits2 = clf(z2)
            loss2 = criterion(logits2, y)
            grads = torch.autograd.grad(loss2, z2, grad_outputs=torch.ones_like(loss2), retain_graph=False)[0]
            abs_grads = grads.abs().mean(dim=0).cpu()
            grad_sums += abs_grads
            grad_counts += 1
            # intervention
            noise = torch.randn_like(z) * args.noise_std
            mask = torch.zeros_like(z)
            mask[:, priority_dims] = 1.0
            z_pert = z + noise * mask.to(device)
            logits_pert = clf(z_pert)
            loss_inv = criterion(logits_pert, y)
            loss_sens = mse(logits, logits_pert)
            loss = loss_cls + loss_inv + args.lambda_sens * loss_sens
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # metrics
            running_loss += loss_cls.item() * y.size(0)
            preds = logits.argmax(dim=1)
            running_corr += (preds == y).sum().item()
            running_tot += y.size(0)
        train_loss = running_loss / running_tot
        train_acc = running_corr / running_tot
        # update priority dims
        avg_grads = grad_sums / grad_counts
        priority_dims = torch.argsort(avg_grads, descending=True)[:k].tolist()
        grad_sums.zero_(); grad_counts = 0
        # validation
        encoder.eval(); clf.eval()
        v_loss = v_corr = v_tot = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                z = encoder(x); logits = clf(z)
                loss = criterion(logits, y)
                v_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                v_corr += (preds == y).sum().item()
                v_tot += y.size(0)
        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        logging.info(f"AIFS Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc})
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    hist_path = os.path.join(args.output_dir, 'aifs_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    # evaluate on test set (distribution shift)
    test_ds = ColoredMNIST(root=args.data_dir, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    encoder.eval(); clf.eval()
    t_loss = 0; t_corr = 0; t_tot = 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x); logits = clf(z)
            loss = criterion(logits, y)
            t_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            t_corr += (preds == y).sum().item()
            t_tot += y.size(0)
    test_loss = t_loss / t_tot
    test_acc = t_corr / t_tot
    logging.info(f"AIFS Test: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    test_path = os.path.join(args.output_dir, 'aifs_test.json')
    with open(test_path, 'w') as f:
        json.dump({'test_loss': test_loss, 'test_acc': test_acc}, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='~/data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--val_split', type=float, default=0.1)
    # aifs params
    parser.add_argument('--k_dims', type=int, default=16)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--lambda_sens', type=float, default=1.0)
    parser.add_argument('--method', type=str, choices=['ERM', 'AIFS'], default='ERM')
    args = parser.parse_args()
    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.method == 'ERM':
        train_erm(args)
    else:
        train_aifs(args)

if __name__ == '__main__':
    main()
