# Complex-Valued Neural Networks: Robustness to Neuron Damage

## Research Question

**Are complex-valued neural networks more robust to neuron damage than real-valued neural networks?**

## Scientific Approach

This project conducts rigorous scientific validation of the hypothesis that complex-valued neural networks exhibit superior robustness to structural damage (neuron removal) compared to real-valued counterparts.

### Key Principles

1. **Fair Comparison**: Complex and real models matched by parameter count
   - 1 complex neuron = 2 real neurons (real part + imaginary part)

2. **Statistical Validation**: All experiments repeated 5+ times
   - Results reported as mean ± standard deviation
   - Statistical significance testing (p < 0.05)

3. **Reproducibility**: Fixed random seeds, documented hyperparameters

4. **Honest Reporting**: Results reported as-is, regardless of hypothesis support

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── config.py              # Hyperparameters, seeds
├── models/
│   ├── complex_layers.py  # ComplexLinear, ComplexConv2d, complex_relu
│   ├── real_models.py     # RealMLP, RealCNN
│   └── complex_models.py  # ComplexMLP, ComplexCNN
├── experiments/
│   ├── exp1_baseline.py   # Baseline performance comparison
│   ├── exp2_damage.py     # Damage robustness
│   └── exp3_layer_analysis.py
├── utils/
│   ├── data.py            # Data loading
│   ├── metrics.py         # Evaluation functions
│   └── visualization.py   # Graph generation
└── results/
    ├── figures/
    └── tables/
```

## Literature Foundation

Based on:
1. **Trabelsi et al. (ICLR 2018)**: Deep Complex Networks implementation
2. **Guberman (2016)**: Complex CNNs show "significantly less vulnerability to overfitting"
3. **Arjovsky et al. (2016)**: Unitary matrices preserve gradients
4. **Nguyen et al. (2015)**: Neural network robustness problem definition
5. **Gal & Ghahramani (2016)**: Dropout and uncertainty

## Experiments

### Experiment 1: Baseline Performance
- Datasets: MNIST, CIFAR-10
- Models: RealMLP vs ComplexMLP, RealCNN vs ComplexCNN
- Verify: Parameter counts equal, baseline accuracy comparable

### Experiment 2: Neuron Damage Robustness
- Damage rates: 0%, 10%, 20%, ..., 90%
- Method: Random neuron masking at inference time
- Analysis: Accuracy degradation curves with error bars

### Experiment 3: Layer-wise Damage Analysis
- Test: Damage individual layers vs all layers
- Find: Which layers are most critical
- Compare: Robustness patterns between model types

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run baseline experiments
python experiments/exp1_baseline.py

# Run damage robustness tests
python experiments/exp2_damage.py

# Run layer-wise analysis
python experiments/exp3_layer_analysis.py
```

## Expected Outputs

1. **Performance Tables**: Baseline accuracy, parameter counts
2. **Robustness Graphs**: Damage rate vs accuracy with error bars
3. **Statistical Tests**: p-values, confidence intervals
4. **Final Report**: 1-page summary with conclusions and limitations

## What This Project Does NOT Claim

- ❌ No connection to quantum mechanics
- ❌ No claims about AGI or brain-like computation
- ❌ No overstated "revolutionary" language
- ✅ Focused, testable scientific hypothesis

## Citation

If you use this code, please cite the foundational papers listed above.

## License

MIT License - For research and educational purposes
