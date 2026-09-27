# Research program: interference, coherence and fault tolerance

This directory is an autonomous research loop in the style of Karpathy's *autoresearch*: a fixed
protocol, a mutable experiment harness (`lab/`), an append-only results file (`runs/runs.jsonl`)
and a chronological log (`LOG.md`) of every hypothesis, run and keep/discard decision. The paper
draft lives in `PAPER.md`.

## Status (2026-09-28)

Phase 1 is complete: experiments E1-E5 ran (369 logged runs plus pilots and the outlier study) and
the paper draft is in [`PAPER.md`](PAPER.md). Headline findings, all confirmed on held-out seeds:

1. A first-order identity, `E||delta||^2 / ||Mu||^2 = p^2 + p(1-p) rho`, holds exactly, and `rho`
   measured on clean data ranks unit-death and weight-noise robustness across real, complex and
   split-complex networks (Spearman -0.71 / -0.84; -0.84 / -0.90 on a different task).
2. TurboQuant / QuaRot-style rotation helps only when faults are aligned with high-gain directions:
   the predicted crossover under function-preserving channel outliers matches the measured one;
   rotation always hurts erasures, and is catastrophic when dead ReLU units keep BatchNorm gain.
3. Penalising `rho` is a Goodhart trap; centred penalties trade activation-quantization robustness
   for unit-death robustness; dropout improves every fault family.
4. Complex-valued networks are not more robust on MNIST once initialisation is matched (the
   original demo's fragility was an initialisation artifact). On unknown-phase tone detection, the
   phase-symmetric complex design is +2.1 points more accurate than a parameter-matched real network
   and far more fault tolerant (R 0.868 vs 0.780); removing the complex algebra, the phase-preserving
   activation or the intensity readout removes the advantage.

Some predictions below were refuted (P4 for quantization, P5 as stated); `LOG.md` records how.

## Vision

Neural networks are moving onto unreliable physical substrates: photonic chips that compute with
complex light amplitudes, analog in-memory hardware, aggressively quantized accelerators. On such
hardware units die, weights drift and every number is quantized. The question is not only *how
accurate* a network is, but *how gracefully it degrades*.

The project started from one hypothesis, *complex-valued networks are more robust to neuron damage
than real-valued ones*. A reduced demo and two pilots (see `LOG.md`, iterations 0-2) showed the
opposite on MNIST, and traced the gap to a single design choice: the complex network's magnitude
(|z|, "intensity detection") readout. Swapping the complex algebra for split-complex numbers or
changing the damage granularity did not matter.

That observation suggests a more general theory, which this program tests.

## Theory: fault sensitivity = concentration x cancellation

Take any layer whose outputs `u` (n units) feed an affine map `s = M u + c` (eval-mode BatchNorm
folded into the next linear layer). Write the signal as a sum of unit contributions
`M u = sum_j a_j`, `a_j = M[:, j] u_j`.

* **Unit death** (each unit dead with probability p) gives a relative error in `s`
  `E||delta||^2 / ||Mu||^2 = p^2 + p(1-p) rho`, with `rho = sum_j ||a_j||^2 / ||sum_j a_j||^2`.
* `rho` factorises exactly as `rho = kappa_conc * kappa_cancel`:
  * `kappa_cancel = (||M||_F^2 ||u||^2 / n) / ||Mu||^2` compares the actual signal with what a
    randomly oriented input of the same norm would produce. It is 1 when contributions add like
    random vectors, below 1 for constructive interference and above 1 when the output is built
    from large contributions that cancel (**destructive interference**).
  * `kappa_conc = n sum_j ||M_j||^2 u_j^2 / (||M||_F^2 ||u||^2)` measures how unevenly the work is
    spread over units (1 = even).
* **Activation quantization** (step proportional to max|u|) gives relative error
  `proportional to crest(u)^2 * kappa_cancel`, crest^2 = n max_j u_j^2 / ||u||^2.
* **Weight quantization / noise** gives the analogous product of a weight-outlier factor and a
  cancellation factor computed on the same layer.

So every fault type is (fault size) x (a fault-specific *concentration* factor) x (a shared
*cancellation* factor). Random rotations, the core of TurboQuant / QuaRot style incoherence
processing, make the concentration factor ~1 but leave the cancellation factor unchanged, because
`||M R^T||_F = ||M||_F`, `||R u|| = ||u||` and `M R^T R u = M u`.

Physically, cancellation is destructive interference. Coherent optical hardware computes *by*
interference, and intensity detection (|z|) forces a network to cancel fields to express "no
evidence for this class". The theory therefore predicts that intensity readouts are fragile to
every fault type at once, and that the fix is architectural, not a rotation.

## Falsifiable predictions

* **P1** `rho` (measured on clean data, no fault sampling) predicts end-to-end robustness to unit
  death across architectures, widths and training choices.
* **P2** Complex networks with intensity (|z|) readouts have high `kappa_cancel` at the readout,
  concentrated on wrong-class outputs.
* **P3** Rotation coding of the physical units (TurboQuant-style incoherence) reduces unit-death
  sensitivity when `kappa_conc > 1`, and cannot remove the cancellation gap.
* **P4** The same `kappa_cancel` ranks models for weight quantization, activation quantization and
  weight noise: fragility is shared across fault types.
* **P5** Interventions that lower cancellation (coherent/linear readout, dropout, an explicit
  cancellation penalty) raise robustness to all fault types together; interventions that only
  lower concentration (rotations) help quantization and channel faults only.
* **P6** On complex-native data (I/Q signals with random carrier phase) complex networks with a
  phase-invariant intensity readout gain clean accuracy; the program asks whether that gain can be
  kept without the fragility.

Any of these can fail; failures are logged and reported as findings.

## Protocol (fixed; changes only via a logged decision)

| Item | Setting |
| --- | --- |
| Data | MNIST (search), Fashion-MNIST (confirmation), synthetic I/Q modulation (complex-native, E4). Repo split: 54k train / 6k val (seed 42) / 10k test |
| Models | MLPs with hidden algebra real / complex / split-complex, readout linear / modulus (algebra-structured) / modulus_untied, ReLU (CReLU) or modReLU, BatchNorm after the activation (as in the original repo). All effective weights start from the same distribution, U(-1/sqrt(fan_in), 1/sqrt(fan_in)); biases start at 0 |
| Training | Adam, lr 1e-3, weight decay 1e-4, batch 128, 5 epochs, no early stopping |
| Seeds | search seeds 0, 1, 2; confirmation seeds 3, 4, 5 (never used for decisions) |
| Unit death | all hidden layers, rates 0.1-0.9, exact counts (randperm), 10 masks per rate; "natural" granularity (real units; complex/split pairs) and per-real-unit granularity |
| Weight noise | multiplicative Gaussian on every stored weight parameter, sigma 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 5 draws |
| Weight quantization | symmetric per-row uniform on stored parameters, 8, 6, 5, 4, 3, 2 bits; plus rotated variant |
| Activation quantization | asymmetric per-tensor uniform at every hidden site, calibrated on 2k training images, 8, 6, 4, 3, 2 bits; plus rotated variant |
| Score | retained accuracy = accuracy under fault / clean accuracy, averaged over the fault grid, per fault family; robustness score R = mean over families |
| Mechanism metrics | per hidden site: `rho`, `kappa_conc`, `kappa_cancel` (natural and unit granularity), signal-relative and full-activation-relative; readout cancellation split by correct vs wrong classes |

## Loop rules

1. One hypothesis per iteration, stated before the run, with the result that would refute it.
2. Every run appends to `runs/runs.jsonl` (config, seed, git commit, metrics). Nothing is deleted.
3. A design change is **kept** only if it raises R (or the targeted family) by more than one
   between-seed standard deviation on the search seeds without lowering clean accuracy by more
   than 0.5 points; otherwise it is **discarded** and the log says so.
4. Headline claims are re-run on the confirmation seeds and on Fashion-MNIST before they enter
   `PAPER.md`.
5. Negative and null results are reported.

**Stopping criteria:** experiments E1-E5 below are complete and `PAPER.md` is drafted, or 30 loop
iterations, or the owner stops the loop.

| Experiment | Question |
| --- | --- |
| E1 | Model zoo on MNIST: does coherence (P1, P2, P4) predict robustness? |
| E2 | TurboQuant-style rotations for quantization and unit death (P3) |
| E3 | Interventions that lower cancellation (P5) |
| E4 | Complex-native I/Q task (P6) |
| E5 | Autoresearch hill-climb over the design space, confirmed on held-out seeds and Fashion-MNIST |

## Running

```bash
pip install -r requirements.txt
python research/tests/sanity.py                       # harness checks (CPU, ~1 min, downloads MNIST)
python research/run.py --name real_linear --algebra real --readout linear --seeds 0 1 2
python research/analyze.py                            # tables from runs/runs.jsonl
```

Everything runs on a laptop CPU; one configuration with 3 seeds takes about a minute.
