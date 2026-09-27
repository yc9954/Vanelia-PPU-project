<h1 align="center">Complex-Valued Network Robustness</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch%202.0%2B-torchvision-4493F8?style=flat" alt="PyTorch 2.0+, torchvision" />
  <img src="https://img.shields.io/badge/datasets-MNIST%20%C2%B7%20CIFAR--10-4493F8?style=flat" alt="MNIST, CIFAR-10" />
  <img src="https://img.shields.io/badge/status-framework%20only%2C%20no%20results-4493F8?style=flat" alt="Framework only, no results" />
</p>

<p align="center">
  <strong>Are complex-valued neural networks more robust to neuron damage than real-valued ones?</strong><br/>
  A small PyTorch research framework that pairs complex and real MLPs / CNNs at equal parameter counts,<br/>
  masks neurons at inference time, and measures how accuracy degrades. The models and utilities are implemented;<br/>
  the experiment scripts and results are not.
</p>

<h3 align="center"><a href="#getting-started"><ins>Getting started</ins></a> · <a href="PROJECT_STATUS.md">Project status document</a></h3>

## Features

- **Complex layers from scratch.** `ComplexLinear` implements `(W_r + i·W_i)(x_r + i·x_i)`, plus `ComplexConv2d`, `ComplexBatchNorm1d/2d`, `ComplexMaxPool2d`, `complex_relu` (ReLU on real and imaginary parts separately), `complex_modulus`, and complex Glorot initialisation following Trabelsi et al.
- **Matched model pairs.** `RealMLP` 784 → 64 → 64 → 10 vs `ComplexMLP` 784 → 32c → 32c → 10 for MNIST; `RealCNN` 3 → 32 → 64 → FC128 → 10 vs `ComplexCNN` 3 → 16c → 32c → FC64c → 10 for CIFAR-10. One complex neuron counts as two real parameters, and `verify_parameters.py` checks the counts line up.
- **Damage as a forward hook.** `apply_neuron_damage()` builds a random mask for a given damage rate (optionally for one layer), `DamageHook` applies it during inference, and `evaluate_with_damage()` repeats the evaluation over several masks and reports mean ± std.
- **Data and plots ready.** MNIST and CIFAR-10 loaders with a 90/10 train/validation split and CIFAR augmentation; plotting helpers for robustness curves with error bars, parameter comparison, layer-wise analysis, training curves, and markdown / CSV result tables.
- **Pre-registered protocol.** `config.py` fixes seed 42, 5 trials per experiment, 50 epochs with Adam (lr 1e-3, weight decay 1e-4, early stopping), damage rates 0 % to 90 % in 10 % steps with 10 masks per rate, a 50 % rate for the layer-wise study, and p < 0.05 for significance.

**Literature foundation** (preserved from the original write-up)

1. Trabelsi et al. (ICLR 2018), *Deep Complex Networks*: the layer implementation.
2. Guberman (2016): complex CNNs show "significantly less vulnerability to overfitting".
3. Arjovsky et al. (2016): unitary matrices preserve gradients.
4. Nguyen et al. (2015): the neural-network robustness problem definition.
5. Gal & Ghahramani (2016): dropout and uncertainty, the link between masking and damage simulation.

The identified gap: no prior systematic study of *structural* (neuron-removal) damage in complex-valued networks; existing work covers overfitting, adversarial robustness and gradient stability.

---

## How it works

```text
config.py (seeds, hyperparameters, damage rates)
     │
     ▼
utils/data.py ──► MNIST / CIFAR-10 loaders (train 90 / val 10 / test)
     │
     ▼
models/  RealMLP ◄─ matched params ─► ComplexMLP        (MNIST)
         RealCNN ◄─ matched params ─► ComplexCNN        (CIFAR-10)
     │
     ▼
utils/metrics.py  train_one_epoch → evaluate_model
                  apply_neuron_damage(rate, layer) → DamageHook → evaluate_with_damage (n trials)
     │
     ▼
utils/visualization.py  robustness curves · parameter bars · layer analysis · tables
```

1. **Match.** Complex models are sized so that their effective real parameter count equals the real model's (`count_parameters` counts complex weights twice).
2. **Train.** Both models of a pair are trained with the same optimiser, schedule and seed, repeated across trials.
3. **Damage.** At inference, a random subset of neurons in one or all layers is masked to zero; each damage rate is evaluated over several independent masks.
4. **Compare.** Accuracy-vs-damage curves with error bars, per-layer sensitivity, and t-tests between complex and real models at each rate.

The three planned experiments: **Exp 1** baseline performance and parameter parity; **Exp 2** damage robustness across 0–90 %; **Exp 3** layer-wise damage (layer 1 only, layer 2 only, all layers at 50 %).

---

## Tech stack

<p>
  <kbd>Python</kbd> &nbsp; <kbd>PyTorch&nbsp;2.0+</kbd> &nbsp; <kbd>torchvision&nbsp;0.15+</kbd> &nbsp; <kbd>NumPy</kbd> &nbsp; <kbd>SciPy</kbd> &nbsp; <kbd>scikit-learn</kbd> &nbsp; <kbd>pandas</kbd> &nbsp; <kbd>matplotlib</kbd> &nbsp; <kbd>seaborn</kbd> &nbsp; <kbd>tqdm</kbd> &nbsp; <kbd>TensorBoard</kbd>
</p>

---

## Getting started

**Prerequisites**

- Python 3 with pip. A GPU is used automatically when CUDA is available (`config.DEVICE`); CPU works.
- MNIST and CIFAR-10 download to `./data` on first use through torchvision.

```bash
git clone https://github.com/yc9954/Vanelia-PPU-project.git
cd Vanelia-PPU-project
pip install -r requirements.txt

python test_installation.py     # imports, model construction, forward passes on both datasets
python verify_parameters.py     # confirms complex and real models have matching parameter counts
```

The commands `python experiments/exp1_baseline.py`, `exp2_damage.py` and `exp3_layer_analysis.py` from the original README do not exist yet; `experiments/` contains only an `__init__.py`. The building blocks to write them are in `utils/metrics.py` (`train_one_epoch`, `evaluate_model`, `evaluate_with_damage`) and `utils/visualization.py`.

---

## Repository structure

| Path | What lives there |
| --- | --- |
| `config.py` | Seeds, datasets, training hyperparameters, MLP / CNN configs, damage rates, device, output dirs, statistics settings. |
| `models/complex_layers.py` | `ComplexLinear`, `ComplexConv2d`, `ComplexBatchNorm1d/2d`, `ComplexMaxPool2d`, `complex_relu`, `complex_modulus`, `complex_flatten`. |
| `models/real_models.py`, `models/complex_models.py` | `RealMLP`, `RealCNN`, `ComplexMLP`, `ComplexCNN`, each with `get_layer_activations`; parameter counters. |
| `utils/data.py` | MNIST and CIFAR-10 dataloaders and dataset info. |
| `utils/metrics.py` | Parameter counting, evaluation, neuron-damage masks and hook, multi-trial damaged evaluation, one training epoch. |
| `utils/visualization.py` | Robustness curves, parameter comparison, layer analysis, training curves, results tables. |
| `experiments/` | Empty package; the three experiment scripts are planned here. |
| `test_installation.py`, `verify_parameters.py` | Smoke test and parameter-parity check. |
| `PROJECT_STATUS.md` | The original task checklist and expected result formats (dated 2025-12-26). |

---

## Project status

An unfinished research scaffold, generated in one session on 2025-12-26 (branch `claude/complex-neural-networks-robustness-6X7ac`, merged in PR #1). Stated plainly:

**Working today.** The complex layers, the four models, the data loaders, the damage-evaluation utilities and the plotting helpers, plus the two verification scripts.

**Not done.** None of the three experiments has been written or run. There are no trained models, no results, no figures and no statistical tests in the repository. The `results/` directory from the original layout does not exist. The accuracy numbers in `PROJECT_STATUS.md` are placeholders for the expected table format, not measurements.

**Scope, as the original write-up put it.** No claims about quantum mechanics, AGI or brain-like computation; one focused, testable hypothesis, with results to be reported as-is whether or not they support it.

**Note on the name.** The repository is called `Vanelia-PPU-project`, but everything in it is the complex-valued robustness study described above.

---

## License

The original README stated "MIT License", but no LICENSE file is committed yet, so default copyright applies: all rights reserved.
