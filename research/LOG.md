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
