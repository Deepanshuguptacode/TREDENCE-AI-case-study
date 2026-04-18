"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network whose weights are gated by learnable
sigmoid-based "gate" parameters.  An L1 sparsity penalty drives most
gates toward exactly zero, effectively pruning those connections during
training — no post-training pruning step required.

Author : AI Engineer Challenge Solution
Dataset: CIFAR-10  (downloaded automatically via torchvision)
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear Layer
# ──────────────────────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates a learnable
    gate_scores tensor (same shape as weight) with every connection.

    Forward pass
    ─────────────
    gates       = sigmoid(gate_scores)          ∈ (0, 1)
    pruned_w    = weight ⊙ gates                element-wise product
    output      = x @ pruned_w.T + bias

    Because both weight and gate_scores are registered nn.Parameters,
    gradients flow through both via standard autograd — no custom
    backward pass needed.

    Pruning criterion (applied only at evaluation / reporting time)
    ──────────────────────────────────────────────────────────────
    A connection is considered "pruned" when  sigmoid(gate_score) < threshold
    (default threshold = 1e-2).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias — same initialisation as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: one scalar per weight element, initialised near 0
        # so that sigmoid(gate_score) ≈ 0.5 at the start of training
        # (uninformative prior — the network decides what to keep).
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        """Kaiming uniform for weight (same as nn.Linear); zeros for gates."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Fan-in for bias init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        # Gate scores start at 0  →  sigmoid(0) = 0.5 (neutral)
        nn.init.zeros_(self.gate_scores)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – turn raw scores into gates ∈ (0,1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2 – element-wise gate application
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3 – linear transform  (implemented from scratch with matmul)
        #   x shape  : (batch, in_features)
        #   output   : (batch, out_features)
        return x.matmul(pruned_weights.t()) + self.bias

    # ── helper: return gate values as a flat tensor ───────────────────────────
    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()

    # ── helper: sparsity fraction at inference time ───────────────────────────
    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        g = self.gate_values()
        return (g < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"params=weight+gate_scores+bias")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Network Definition
# ──────────────────────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Four-hidden-layer feed-forward network for CIFAR-10 classification.
    Every linear connection is implemented via PrunableLinear.

    Input : 32×32×3 image flattened to 3072 floats
    Output: 10 class logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            PrunableLinear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            PrunableLinear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(x))

    # ── sparsity helpers ──────────────────────────────────────────────────────
    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.
        Minimising this term drives gates toward 0 (pruned connections).
        """
        total = torch.tensor(0.0, device=DEVICE)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    @torch.no_grad()
    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below threshold."""
        pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.gate_values()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total > 0 else 0.0

    @torch.no_grad()
    def all_gate_values(self) -> np.ndarray:
        """Concatenate all gate values from every PrunableLinear."""
        parts = [layer.gate_values().numpy() for layer in self.prunable_layers()]
        return np.concatenate(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def build_dataloaders(batch_size: int = 128, data_root: str = "./data"):
    """Returns (train_loader, test_loader)."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(data_root, train=True,
                                             download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(data_root, train=False,
                                             download=True, transform=test_tf)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, lambda_sparse: float):
    model.train()
    running_cls_loss = running_sparse_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits  = model(images)

        # Classification loss
        cls_loss    = F.cross_entropy(logits, labels)

        # Sparsity regularisation  (L1 on sigmoid gates)
        sparse_loss = model.sparsity_loss()

        # Total loss
        loss = cls_loss + lambda_sparse * sparse_loss
        loss.backward()

        # Clip gradients for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_cls_loss    += cls_loss.item()
        running_sparse_loss += sparse_loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return (running_cls_loss / n,
            running_sparse_loss / n,
            correct / total)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(lambda_sparse: float,
                   train_loader,
                   test_loader,
                   epochs: int = 30,
                   lr: float = 1e-3):
    """
    Train a fresh SelfPruningNet for `epochs` epochs with the given λ.
    Returns a dict with training history and final metrics.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lambda_sparse}   |   epochs = {epochs}   |   lr = {lr}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(DEVICE)

    # Adam updates both weight and gate_scores simultaneously
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"cls_loss": [], "sparse_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        cls_l, sp_l, tr_acc = train_one_epoch(
            model, train_loader, optimizer, lambda_sparse)
        te_acc = evaluate(model, test_loader)
        scheduler.step()

        history["cls_loss"].append(cls_l)
        history["sparse_loss"].append(sp_l)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            sparsity = model.overall_sparsity()
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"cls={cls_l:.4f}  sp={sp_l:.1f}  "
                  f"train={tr_acc:.3f}  test={te_acc:.3f}  "
                  f"sparsity={sparsity:.1%}")

    final_acc      = evaluate(model, test_loader)
    final_sparsity = model.overall_sparsity()
    gate_vals      = model.all_gate_values()

    print(f"\n  ► Final test accuracy : {final_acc:.4f}  ({final_acc*100:.2f}%)")
    print(f"  ► Overall sparsity   : {final_sparsity:.4f}  ({final_sparsity*100:.2f}%)")

    return {
        "lambda"   : lambda_sparse,
        "model"    : model,
        "history"  : history,
        "test_acc" : final_acc,
        "sparsity" : final_sparsity,
        "gate_vals": gate_vals,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_gate_distribution(gate_vals: np.ndarray,
                           lambda_val: float,
                           save_path: str):
    """
    Histogram of final gate values for a trained model.
    A successful pruning shows a large spike at 0 and another cluster > 0.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gate_vals, bins=100, color="#3A7BD5", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Gate Value  (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distribution of Gate Values  (λ = {lambda_val})\n"
        f"Spike at 0 → pruned connections, cluster > 0 → retained connections",
        fontsize=11)
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2,
               label="prune threshold (0.01)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved gate distribution plot → {save_path}")


def plot_training_curves(results: list, save_path: str):
    """Plot test accuracy over epochs for all λ values on one chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for r in results:
        lam   = r["lambda"]
        label = f"λ={lam}"
        axes[0].plot(r["history"]["test_acc"], label=label)
        axes[1].plot(r["history"]["sparse_loss"], label=label)

    axes[0].set_title("Test Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Sparsity Loss vs Epoch  (L1 of gates)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Sparsity Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR = "./outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = build_dataloaders(batch_size=128)

    # ── three λ values (low / medium / high) ─────────────────────────────────
    LAMBDAS = [1e-5, 1e-4, 5e-4]
    EPOCHS  = 30          # increase to 50–60 for even better results

    all_results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, train_loader, test_loader, epochs=EPOCHS)
        all_results.append(res)

    # ── gate distribution plot for best model ─────────────────────────────────
    # "best" = highest test accuracy
    best = max(all_results, key=lambda r: r["test_acc"])
    plot_gate_distribution(
        best["gate_vals"],
        best["lambda"],
        save_path=os.path.join(OUT_DIR, f"gate_dist_best_lambda{best['lambda']}.png"),
    )

    # ── training curves for all λ ─────────────────────────────────────────────
    plot_training_curves(
        all_results,
        save_path=os.path.join(OUT_DIR, "training_curves.png"),
    )

    # ── gate distribution plots for every λ ──────────────────────────────────
    for r in all_results:
        plot_gate_distribution(
            r["gate_vals"],
            r["lambda"],
            save_path=os.path.join(OUT_DIR, f"gate_dist_lambda{r['lambda']}.png"),
        )

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print(f"  {'Lambda':>10} | {'Test Acc (%)':>12} | {'Sparsity (%)':>12}")
    print("="*55)
    for r in all_results:
        print(f"  {r['lambda']:>10.0e} | "
              f"{r['test_acc']*100:>12.2f} | "
              f"{r['sparsity']*100:>12.2f}")
    print("="*55)

    # ── save results as JSON for the report ───────────────────────────────────
    summary = [
        {"lambda": r["lambda"],
         "test_acc_pct": round(r["test_acc"]*100, 2),
         "sparsity_pct": round(r["sparsity"]*100, 2)}
        for r in all_results
    ]
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll outputs saved to ./outputs/")
    print("Done ✓")
