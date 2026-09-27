# Research log

Chronological. Each iteration states the hypothesis before the run, the setup, the result and the
decision. Numbers are normalised AUC / retained accuracy unless stated otherwise.

---

## Iteration 0: reproduce the repo's reduced demo (2026-09-27)

**Goal.** Check that `experiments/quick_demo.py` (MNIST, 3 epochs, 5 Bernoulli masks, seed 42)
reproduces the README on a different CPU (x86 container vs Apple M5).

**Result.** Every damage rate within 1-3 points of the README (50 % damage: RealMLP 47.3 vs 46.6,
ComplexMLP 25.3 vs 25.0). Direction identical: the complex MLP degrades faster.

**Side findings (code review, verified by running).**
- `utils/metrics.apply_neuron_damage` masks each complex layer three times (parent + `fc_r` +
  `fc_i`), masks output layers, selects layers by the first digit in the module name (silently
  selecting nothing or the wrong layers) and reseeds the global RNGs. The demo bypasses it.
- In the CNNs (conv -> BN -> ReLU) masking the conv output leaves a constant ReLU(beta - gamma mu /
  sigma): on MNIST-trained CNNs 45/64 (real) and 35/64 (complex) "dead" conv2 channels still fire.
- With real input the first complex layer is an unconstrained real layer; later complex layers are
  real layers with tied weights [[A, -B], [B, A]]; the readout is |z| of paired outputs. A
  re-implementation as a real network matches ComplexMLP to 1.6e-6.

**Decision.** Build a separate, verified harness (`research/lab`) instead of the repo's damage code.

---

## Iteration 1: is the gap caused by paired damage or by the complex algebra? (pilot)

**Hypothesis.** The complex MLP looks fragile because a complex neuron's real and imaginary parts
die together (damage in chunks of two). Refuted if the gap persists when parts die independently.

**Setup.** Repo models (RealMLP 64-64, ComplexMLP 32c-32c, SplitComplexMLP with j^2 = +1), MNIST,
3 epochs, seeds 0-2, 10 exact-count masks per rate, both hidden layers damaged.

| model | per-unit damage | paired damage |
| --- | --- | --- |
| RealMLP | 0.571 +- 0.023 | 0.581 +- 0.023 |
| ComplexMLP | 0.480 +- 0.010 | 0.483 +- 0.013 |
| SplitComplexMLP | 0.488 +- 0.008 | 0.467 +- 0.013 |

**Result.** Refuted. Granularity changes nothing; split-complex behaves like complex, so the
complex algebra itself is not the cause either.

---

## Iteration 2: hidden layers or readout? (pilot, 2x2 factorial)

**Hypothesis.** The gap comes from the complex hidden layers (tied weights, half the free
parameters in layer 2). Refuted if complex hidden layers with an ordinary linear readout match the
real model.

| hidden | linear logits | modulus readout |
| --- | --- | --- |
| real | 0.547 +- 0.012 | 0.568 +- 0.039 (untied 64->20, then |z|) |
| complex | 0.550 +- 0.025 | 0.494 +- 0.017 (original ComplexMLP) |

**Result.** Refuted. Complex hidden layers are fine; an untied modulus readout on real hidden
layers is fine; only the complex (tied) layer + |z| readout is fragile. In both pilots every seed
of a model with that readout scored <= 0.51 and every other seed >= 0.53.

**Decision.** Reframe the project around *why* that readout is fragile. Working theory
(README): fault sensitivity = concentration x cancellation; intensity readouts force destructive
interference. Next: E1 measures the factors directly across a model zoo.

---

## Iteration 3: E1 model zoo (18 configurations x 3 search seeds, MNIST, 5 epochs)

**Hypotheses.** P1 (`rho` predicts unit-death robustness), P2 (complex |z| readouts cancel on wrong
classes), P4 (cancellation is shared across fault types). Also re-tests the pilot claim with the
new harness, where every algebra starts from the same effective weight distribution.

**Setup.** `batches/E1.txt`: algebra (real / complex / split) x readout (linear / tied |z| / untied
|z|), widths 32 / 64 / 128, dropout 0.2, modReLU, no BatchNorm, and the repository's original
complex initialisation (`--init repo`). 54 runs, results in `runs/runs.jsonl` (tag E1).

**Results.**

| config | clean | R | death | weight noise | weight quant | act quant |
| --- | --- | --- | --- | --- | --- | --- |
| real, linear | 97.35 | 0.838 | 0.479 +- 0.021 | 0.931 | 0.956 | 0.984 |
| complex, linear | 97.09 | 0.840 | 0.498 +- 0.022 | 0.923 | 0.954 | 0.984 |
| complex, tied \|z\| | 97.13 | 0.823 | 0.487 +- 0.021 | 0.892 | 0.932 | 0.980 |
| complex, tied \|z\|, repo init | 96.75 | 0.789 | 0.435 +- 0.013 | 0.825 | 0.925 | 0.972 |
| split, tied \|z\| | 96.81 | 0.821 | 0.463 +- 0.010 | 0.897 | 0.943 | 0.981 |
| real, linear, dropout 0.2 | 96.80 | 0.890 | 0.646 +- 0.022 | 0.976 | 0.945 | 0.992 |
| complex, tied \|z\|, dropout 0.2 | 96.61 | 0.887 | 0.660 +- 0.014 | 0.958 | 0.947 | 0.983 |

1. **The pilot's "complex is fragile" was mostly an initialisation artifact.** With equal initial
   weight scale the complex |z| model matches the real model under unit death (0.487 vs 0.479).
   The repository's complex initialisation (N(0, 2/(fan_in+fan_out)) per part; about 3x larger at
   the readout) reproduces the gap (0.435). What survives: tied |z| readouts lose 0.03-0.04 under
   multiplicative weight noise (0.892 / 0.897 vs 0.921-0.931 for linear readouts).
2. **P1 supported.** Mean log `rho` over sites, measured on clean data with no fault sampling,
   vs retained accuracy across all 54 runs (Spearman): unit death -0.71, per-unit death -0.68,
   weight noise -0.84, R -0.67; weight quantization -0.18, activation quantization -0.32.
3. **P2 weakly supported.** Wrong-class `kappa_cancel` at the readout: complex |z| 0.58 vs real
   linear 0.48; repo init 0.92 and no-BN 1.04, which are also the least robust to weight noise
   (0.825, 0.808).
4. **P4 partially.** `kappa_cancel` at the last site predicts weight noise (-0.76) but not
   quantization.
5. **Rotation (preview of E2).** Randomized-Hadamard rotation coding is catastrophic under unit
   death in BatchNorm nets (retained 0.10-0.12). Cause: ReLU units that never fire keep large
   BatchNorm gain and hold > 99 % of the first site's downstream gain `||M||_F^2`
   (`kappa_cancel` ~ 1800, `kappa_conc` ~ 0). Erasures never touch them; rotated faults excite them.
   Pruning never-firing units (exact on calibration data) recovers most of it (0.38-0.63), but
   plain still wins. In nets without dead-unit gain (no BN, modReLU) rotation costs 1-10 %, and
   every such net has `kappa_conc` < 1 (0.73-1.03), as the theory requires for rotation to hurt.
   Rotated activation quantization also hurts (plain quantization keeps ReLU zeros exact); rotated
   weight quantization helps slightly (+0.005 to +0.035). The activation outliers here are
   per-input (norm) outliers, which a rotation preserves (crest^2 3474 -> 2298), unlike the
   per-channel outliers of LLMs that QuaRot / TurboQuant target.
6. Dropout 0.2 is the strongest single change (R 0.887-0.890; unit death 0.65-0.66) and lowers
   `rho` at both sites (site 0: 0.83 -> 0.37), as the theory predicts. Width raises unit-death
   robustness (0.43 / 0.48 / 0.52 at 32 / 64 / 128) and lowers `rho`.

**Decision.** Keep the coherence theory as the organising idea; demote "complex vs real" to a case
study (initialisation artifact, weight-noise gap, complex-native data in E4). Next: E3
(interventions that target `rho` directly) and E4 (I/Q data), running now.

---

## Iteration 4: E3 interventions, first pass (16 configurations x 3 seeds)

**Hypothesis (P5).** Training that lowers `rho` (or `kappa_cancel`) raises robustness to every
fault type. Refuted if the penalised networks are not more robust.

| intervention | clean | R | unit death |
| --- | --- | --- | --- |
| baseline real / complex \|z\| | 97.35 / 97.13 | 0.838 / 0.823 | 0.479 / 0.487 |
| dropout 0.1 / 0.2 / 0.3 (real) | 97.11 / 96.80 / 96.33 | 0.881 / 0.890 / 0.896 | 0.605 / 0.646 / 0.686 |
| dropout 0.1 / 0.2 / 0.3 (complex) | 96.95 / 96.61 / 95.78 | 0.879 / 0.887 / 0.892 | 0.614 / 0.660 / 0.701 |
| raw `rho` penalty 0.1 / 0.3 / 1 (real) | 96.83 / 96.09 / 95.02 | 0.740 / 0.665 / 0.593 | 0.313 / 0.193 / 0.143 |
| raw `kappa_cancel` penalty 0.3 / 1 (real) | 96.28 / 94.92 | 0.645 / 0.602 | 0.151 / 0.129 |

**Result.** Dropout works monotonically for both algebras. The raw penalties are refuted in the
strongest way: they drive `rho` to 0.02-0.06, far below any unpenalised network, while robustness
collapses. This is Goodhart's law on a diagnostic: `rho` divides by the raw signal energy
`||Mu||^2`, so a network can inflate an input-independent component of `Mu` that carries no
information. The diagnostic still ranks naturally trained networks (iteration 3), but it is not a
safe training target.

**Decision.** Add centred variants that divide by the variance of `Mu` over inputs (`rho_var`,
`kappa_cancel_var`; penalties `rho_var`, `cancel_var`) and re-run (E3b). Re-run E1 as E1b with the
centred metrics and saved models.

---

## Iteration 5: E2, when does TurboQuant-style rotation help? (controlled outliers)

**Hypothesis (P3).** Rotation makes a site fault isotropic. It should help exactly when the plain
fault lands on high-gain directions more than isotropic noise would, and hurt otherwise. Refuted
if the predicted rotated/plain error ratio does not track which of the two wins.

**Setup.** `e2_outliers.py`. Trained real (BN), complex |z| (BN) and real (no BN) MLPs, 3 seeds.
A function-preserving rescaling multiplies one unit (one complex unit) per hidden site by s = 1, 4,
16, 64: the preceding layer's row and bias times s (ReLU is positively homogeneous), BatchNorm
running statistics rescaled, or the next layer's column divided by s without BatchNorm. Maximum
logit drift <= 1.2e-4, so the function is unchanged; only the activation geometry changes. First-
order prediction for activation quantization:
`rot/plain = range_rot^2 ||M_alive||_F^2 / (range^2 sum_j ||M_j||^2 P(u_j != 0))`.

| model | s | predicted rot/plain (sites 1, 2) | 4-bit acts plain -> rotated | 30 % units dead plain -> rotated |
| --- | --- | --- | --- | --- |
| real, BN | 1 | 7.7, 5.6 | 97.0 -> 96.1 | 74.4 -> 59.3 |
| real, BN | 4 | 1.7, 1.2 | 96.1 -> 95.8 | 74.4 -> 54.7 |
| real, BN | 16 | 0.39, 0.21 | 60.3 -> 88.7 | 74.4 -> 39.3 |
| real, BN | 64 | 0.24, 0.12 | 10.0 -> 42.9 | 74.4 -> 27.3 |
| complex \|z\|, BN | 4 | 1.3, 0.61 | 95.1 -> 95.2 | 69.8 -> 46.6 |
| complex \|z\|, BN | 16 | 0.59, 0.27 | 33.8 -> 68.8 | 69.8 -> 19.6 |
| real, no BN | 16 | 0.21, 1.67 | 63.1 -> 92.3 | 76.8 -> 48.0 |
| real, no BN | 64 | 0.12, 1.62 | 10.4 -> 60.6 | 76.8 -> 32.0 |

**Result.** Supported. The predicted ratio crosses 1 where the measured winner flips (a near tie
at s = 4, rotation far ahead from s = 16). Plain unit death is exactly invariant to the rescaling
(erasures see the same function), while rotated unit death degrades with s because the rotation
leaks the outlier's energy into every unit. Figure: `figures/rotation_crossover.png`.

**Interpretation.** Incoherence processing (QuaRot, TurboQuant) is not a free lunch. It helps
quantization when outliers sit in channels (as in LLMs), is neutral-to-harmful in networks whose
activations are sparse or whose idle units carry gain, and is harmful for erasure-type faults.
One number, the alignment between a fault's covariance and the downstream gain, decides which.
