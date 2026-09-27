"""
Reduced-protocol demo of the three planned experiments on MNIST (and optionally CIFAR-10).

This is NOT the pre-registered protocol in config.py (50 epochs, 5 trials, 10 masks per
rate). It trains each model pair for a few epochs on CPU and produces the figures in
docs/ so the pipeline can be seen end to end. Run the full protocol before drawing
conclusions.

Usage:
    python experiments/quick_demo.py                  # MNIST MLP pair, 3 epochs
    python experiments/quick_demo.py --cifar          # also CIFAR-10 CNN pair
    python experiments/quick_demo.py --epochs 5 --masks 10 --out results/quick_demo
"""

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from models import ComplexCNN, ComplexMLP, RealCNN, RealMLP  # noqa: E402
from utils.data import get_dataloaders  # noqa: E402
from utils.metrics import (  # noqa: E402
    DamageHook,
    count_parameters,
    evaluate_model,
    evaluate_with_damage,
    train_one_epoch,
)
from utils.visualization import (  # noqa: E402
    plot_layer_analysis,
    plot_parameter_comparison,
    plot_robustness_curves,
    plot_training_curves,
)


def train(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_hist = {"loss": [], "accuracy": []}
    val_hist = {"loss": [], "accuracy": []}
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate_model(model, val_loader, device, criterion)
        train_hist["loss"].append(tr["loss"])
        train_hist["accuracy"].append(tr["accuracy"])
        val_hist["loss"].append(va["loss"])
        val_hist["accuracy"].append(va["accuracy"])
        print(
            f"  epoch {epoch + 1}/{epochs}  train {tr['accuracy']:.2f}%  "
            f"val {va['accuracy']:.2f}%  ({time.time() - t0:.0f}s)"
        )
    return train_hist, val_hist


def hidden_layer_modules(model):
    """Return [(name, module)] for the trainable hidden layers, in forward order.

    utils.metrics.apply_neuron_damage() is not used here for two reasons: its layer_idx
    parses a number out of the module name (fc1 -> 0), which does not match the names
    these models use ('network.0', 'hidden_layers.0', ...), and with layer_idx=None it
    masks the nn.Linear/Conv2d children (fc_r, fc_i, conv_r, conv_i) of every complex
    layer in addition to the complex layer itself, and the output layer of both models.
    """
    layers = []
    for name, module in model.named_modules():
        if name == "" or name.rsplit(".", 1)[-1] in ("fc_r", "fc_i", "conv_r", "conv_i"):
            continue  # the real sub-layers inside a complex layer are masked through their parent
        cls = module.__class__.__name__
        if isinstance(module, (nn.Linear, nn.Conv2d)) or cls in ("ComplexLinear", "ComplexConv2d"):
            layers.append((name, module))
    # The last one is the output layer (network.6, output_layer, fc2); the study damages hidden layers.
    return layers[:-1]


def damage_named_layers(model, loader, device, names, rate, masks, seed=config.RANDOM_SEED):
    """Mask `rate` of the output neurons/channels of the named layers, over `masks` draws."""
    accs = []
    for k in range(masks):
        g = torch.Generator().manual_seed(seed + k)
        hooks = []
        for name, module in model.named_modules():
            if name not in names:
                continue
            if isinstance(module, nn.Linear):
                n = module.out_features
            elif isinstance(module, nn.Conv2d):
                n = module.out_channels
            elif module.__class__.__name__ == "ComplexLinear":
                n = module.fc_r.out_features
            else:
                n = module.conv_r.out_channels
            mask = (torch.rand(n, generator=g) > rate).float().to(device)
            hooks.append(module.register_forward_hook(DamageHook(mask)))
        accs.append(evaluate_model(model, loader, device)["accuracy"])
        for h in hooks:
            h.remove()
    return float(np.mean(accs)), float(np.std(accs))


def run_pair(tag, real_model, complex_model, dataset, args, device, out):
    print(f"\n=== {tag}: {dataset} ===")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=config.BATCH_SIZE, num_workers=0, data_dir=config.DATA_DIR,
        seed=config.RANDOM_SEED,
    )
    pair = {"Real" + tag: real_model.to(device), "Complex" + tag: complex_model.to(device)}
    params = {n: count_parameters(m) for n, m in pair.items()}
    print("  parameters:", params)

    histories, robustness, layerwise, summary = {}, {}, {}, {}
    hidden = {n: [nm for nm, _ in hidden_layer_modules(m)] for n, m in pair.items()}
    for name, model in pair.items():
        torch.manual_seed(config.RANDOM_SEED)
        print(f"  training {name} for {args.epochs} epochs")
        histories[name] = train(model, train_loader, val_loader, args.epochs, device)
        clean = evaluate_model(model, test_loader, device)["accuracy"]
        print(f"  {name} clean test accuracy {clean:.2f}%")

        # Exp 2: damage every hidden layer at each rate, `args.masks` random masks each.
        means, stds = [], []
        for rate in config.DAMAGE_RATES:
            m, s = damage_named_layers(model, test_loader, device, hidden[name], rate, args.masks)
            means.append(m)
            stds.append(s)
            print(f"    damage {int(rate * 100):2d}%  {m:6.2f} +- {s:.2f}")
        robustness[name] = {
            "damage_rates": config.DAMAGE_RATES, "mean_accuracies": means, "std_accuracies": stds,
        }

        # Exp 3: 50 % damage on layer 1 only, layer 2 only, all hidden layers.
        l1, l2 = hidden[name][0], hidden[name][1]
        rows = [
            damage_named_layers(model, test_loader, device, [l1], config.LAYER_DAMAGE_RATE, args.masks),
            damage_named_layers(model, test_loader, device, [l2], config.LAYER_DAMAGE_RATE, args.masks),
            damage_named_layers(model, test_loader, device, hidden[name], config.LAYER_DAMAGE_RATE, args.masks),
        ]
        layerwise[name] = {
            "layers": ["Layer 1", "Layer 2", "All Layers"],
            "accuracies": [r[0] for r in rows], "stds": [r[1] for r in rows],
        }
        summary[name] = {"params": params[name], "clean_test_acc": clean,
                         "hidden_layers": hidden[name]}

    # Also run the library's own all-layer damage helper once, for reference.
    for name, model in pair.items():
        r = evaluate_with_damage(model, test_loader, device, 0.5, seed=config.RANDOM_SEED, num_trials=3)
        summary[name]["evaluate_with_damage_50pct_all_modules"] = r["mean"]

    os.makedirs(out, exist_ok=True)
    plot_robustness_curves(
        robustness, save_path=f"{out}/robustness_{tag.lower()}.png",
        title=f"Neuron damage robustness on {dataset} ({args.epochs} epochs, {args.masks} masks per rate)",
    )
    plot_layer_analysis(layerwise, save_path=f"{out}/layerwise_{tag.lower()}.png")
    plot_parameter_comparison(params, save_path=f"{out}/params_{tag.lower()}.png")
    for name in pair:
        plot_training_curves(histories[name][0], histories[name][1],
                             save_path=f"{out}/training_{name.lower()}.png")
    with open(f"{out}/results_{tag.lower()}.json", "w") as f:
        json.dump({"summary": summary, "robustness": robustness, "layerwise": layerwise,
                   "histories": histories, "epochs": args.epochs, "masks": args.masks}, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--masks", type=int, default=5, help="random masks per damage rate")
    p.add_argument("--cifar", action="store_true", help="also run the CNN pair on CIFAR-10")
    p.add_argument("--out", default="results/quick_demo")
    args = p.parse_args()

    device = config.DEVICE
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    print(f"device: {device}")

    mlp = config.MLP_CONFIG
    run_pair("MLP", RealMLP(mlp["real"]["input_size"], mlp["real"]["hidden_sizes"], mlp["real"]["output_size"]),
             ComplexMLP(mlp["complex"]["input_size"], mlp["complex"]["hidden_sizes"], mlp["complex"]["output_size"]),
             "MNIST", args, device, args.out)
    if args.cifar:
        cnn = config.CNN_CONFIG
        run_pair("CNN", RealCNN(cnn["real"]["conv_channels"], cnn["real"]["fc_size"], cnn["real"]["output_size"]),
                 ComplexCNN(cnn["complex"]["conv_channels"], cnn["complex"]["fc_size"], cnn["complex"]["output_size"]),
                 "CIFAR10", args, device, args.out)


if __name__ == "__main__":
    main()
