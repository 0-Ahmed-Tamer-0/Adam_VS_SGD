# Adam vs SGD: Optimizer Comparison on MNIST

A comprehensive deep learning comparison between the **Adam optimizer** and **Stochastic Gradient Descent (SGD)** optimizer for handwritten digit recognition using the MNIST dataset with Keras.

## Overview

This project provides an in-depth empirical analysis of two fundamental optimization algorithms in deep learning. By training neural networks on the MNIST handwritten digit dataset, we compare the convergence behavior, training efficiency, and model performance of both optimizers across various metrics.

**Purpose:** Understand the practical differences between adaptive learning rate methods (Adam) and traditional first-order optimization (SGD) through side-by-side experimentation.

## Project Objectives

1. **Comparative Analysis**: Evaluate both optimizers on identical neural network architectures
2. **Performance Metrics**: Track convergence speed, training time, and accuracy
3. **Visualization**: Generate comparative plots showing learning curves and performance differences
4. **Insights**: Provide practical recommendations for optimizer selection based on empirical results

## Optimizers Compared

### SGD (Stochastic Gradient Descent)
- **Type**: First-order optimization algorithm
- **Learning Rule**: Updates parameters using mini-batch gradients
- **Hyperparameters**: Learning rate, momentum (optional)
- **Characteristics**:
  - Simple and interpretable
  - May require careful learning rate tuning
  - Slower convergence compared to adaptive methods
  - Better generalization in some cases
  - Requires momentum for improved performance

### Adam (Adaptive Moment Estimation)
- **Type**: Adaptive learning rate optimization algorithm
- **Learning Rule**: Combines momentum with adaptive per-parameter learning rates
- **Hyperparameters**: Learning rate, β₁, β₂ (momentum coefficients), ε (numerical stability)
- **Characteristics**:
  - Faster initial convergence
  - Less sensitive to learning rate tuning
  - Combines advantages of momentum and RMSprop
  - Adaptive learning rates per parameter
  - May converge to suboptimal solutions in some cases

## Dataset: MNIST

**MNIST Handwritten Digit Database**
- **Total Images**: 70,000 (60,000 training + 10,000 test)
- **Image Size**: 28×28 pixels
- **Pixel Values**: 0-255 (normalized to 0-1)
- **Classes**: 10 digit categories (0-9)
- **Format**: Grayscale images
- **Source**: Keras built-in dataset

## Methodology

### Neural Network Architecture
- **Layer 1**: Input layer (784 neurons for 28×28 flattened images)
- **Layer 2**: Hidden layer with ReLU activation (typically 128 or 256 units)
- **Layer 3**: Output layer with Softmax activation (10 units for digit classes)
- **Loss Function**: Categorical Cross-entropy

### Experimental Setup
1. **Train-Test Split**: 60,000 training, 10,000 test samples
2. **Batch Size**: Configurable (typically 32 or 128)
3. **Epochs**: Multiple training runs (typically 10-20 epochs)
4. **Validation**: 20% of training data held as validation set
5. **Metrics Tracked**:
   - Training loss
   - Validation loss
   - Training accuracy
   - Validation accuracy
   - Convergence speed
   - Training time per epoch

### Optimizer Configurations

**SGD Variant**
```python
optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0)
```

**Adam Variant**
```python
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
```

## Key Findings

### Convergence Speed
- **Adam**: Typically reaches optimal performance in fewer epochs
- **SGD**: Requires more epochs but may reach better final accuracy

### Training Dynamics
- **Adam**: Smooth, steady convergence with fast initial improvements
- **SGD**: More variable convergence, benefits from momentum

### Generalization
- **SGD**: Often generalizes better to unseen data
- **Adam**: May overfit if not properly regularized

### Computational Efficiency
- **Adam**: Slightly higher memory overhead (maintains exponential moving averages)
- **SGD**: Lower memory requirements

## Technical Stack

### Deep Learning Framework
- **Keras/TensorFlow**: Neural network implementation and optimization
- **TensorFlow 2.x**: Eager execution and Keras API

### Data Processing
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data handling and analysis

### Visualization
- **Matplotlib**: Plot generation and comparison charts
- **Seaborn**: Enhanced statistical visualizations (optional)

### Utilities
- **Jupyter Notebook**: Interactive analysis and documentation
- **Scikit-learn**: Metrics and evaluation

## Installation

### Requirements
```bash
pip install tensorflow keras numpy matplotlib jupyter
```

### Full Setup
```bash
# Clone repository
git clone https://github.com/0-Ahmed-Tamer-0/Adam_VS_SGD.git
cd Adam_VS_SGD

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Usage

### Running the Analysis
1. Open `Adam_VS_SGD.ipynb` in Jupyter Notebook
2. Execute cells sequentially to:
   - Load and preprocess MNIST data
   - Build neural network models
   - Train with SGD optimizer
   - Train with Adam optimizer
   - Generate comparison visualizations
   - Analyze results and insights

### Expected Output
- Training/validation loss curves for both optimizers
- Training/validation accuracy curves
- Side-by-side performance comparison plots
- Statistical summary table
- Recommendations for optimizer selection

## Comparison Visualizations

The notebook generates multiple comparison plots:
1. **Loss Comparison**: Training and validation loss over epochs
2. **Accuracy Comparison**: Training and validation accuracy progression
3. **Convergence Curves**: Rate of convergence for both optimizers
4. **Performance Table**: Summary statistics and final metrics

## Results Summary

### Typical Results on MNIST

| Metric | SGD | Adam |
|--------|-----|------|
| Final Test Accuracy | ~97.5% | ~98.0% |
| Training Time | Longer | Faster |
| Convergence Epochs | 15-20 | 10-15 |
| Training Loss Stability | Variable | Smooth |
| Generalization | Often Better | Good |

## Key Insights & Recommendations

### When to Use SGD
- ✅ When training time is not critical
- ✅ For maximum generalization performance
- ✅ When you need fine-grained learning rate control
- ✅ For reinforcement learning tasks
- ✅ When computational memory is limited

### When to Use Adam
- ✅ For quick prototyping and experimentation
- ✅ When convergence speed is important
- ✅ For most practical deep learning tasks
- ✅ When fine-tuning pre-trained models
- ✅ For complex non-convex optimization problems

### Hybrid Approach
- Train with Adam for fast convergence to good starting point
- Fine-tune with SGD for better generalization

## Project Structure

```
Adam_VS_SGD/
├── Adam_VS_SGD.ipynb          # Main analysis notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── results/                    # (Optional) Output plots and results
    ├── comparison_plots.png
    ├── loss_curves.png
    └── accuracy_curves.png
```

## Notebook Sections

1. **Setup**: Import libraries and configuration
2. **Data Loading**: MNIST dataset loading and exploration
3. **Preprocessing**: Normalization and flattening
4. **Model Definition**: Neural network architecture
5. **SGD Training**: Training with Stochastic Gradient Descent
6. **Adam Training**: Training with Adam optimizer
7. **Evaluation**: Test set performance comparison
8. **Visualization**: Comparative plots and analysis
9. **Insights**: Key findings and recommendations

## Theoretical Background

### Gradient Descent Optimization
- **Batch GD**: Uses all training data (slow, stable)
- **SGD**: Uses one sample (fast, noisy)
- **Mini-batch SGD**: Compromise between speed and stability

### Adaptive Methods
- **Momentum**: Accelerates gradient descent in relevant directions
- **RMSprop**: Adapts learning rate per parameter using moving average of gradients
- **Adam**: Combines momentum and RMSprop advantages

## Hyperparameter Tuning

### SGD Hyperparameters
```python
learning_rate = 0.01      # Controls step size
momentum = 0.9            # Acceleration factor
decay = 0.0001           # Learning rate decay
nesterov = True          # Use Nesterov momentum
```

### Adam Hyperparameters
```python
learning_rate = 0.001    # Initial learning rate
beta_1 = 0.9             # Exponential decay rate for 1st moment
beta_2 = 0.999           # Exponential decay rate for 2nd moment
epsilon = 1e-7           # Numerical stability constant
```

## Extensions & Future Work

- [ ] Compare with other optimizers (RMSprop, Nadam, AdaGrad)
- [ ] Test on larger, more complex datasets
- [ ] Analyze optimizer behavior across network architectures
- [ ] Implement learning rate scheduling
- [ ] Compare convergence on different activation functions
- [ ] Batch size impact analysis
- [ ] Statistical significance testing
- [ ] Visualize parameter updates and gradients

## References

- Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980.
- Ruder, S. (2016). "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747.
- LeCun, Y., Cortes, C., & Burges, C. J. (1998). "The MNIST database of handwritten digits."
- Keras/TensorFlow Documentation: https://keras.io/

## Contributing

Contributions are welcome! Feel free to:
- Add more optimizer comparisons
- Extend analysis to other datasets
- Improve visualizations
- Report findings and insights
- Submit pull requests

## License

This project is open source and available under the MIT License.

## Author

**Ahmed Tamer Mohamed Ezzat**

## Contact

For questions or suggestions, please reach out to the repository owner.

---

**Note:** This project is an educational resource demonstrating practical optimization algorithm comparison in deep learning. Results may vary based on random initialization and hyperparameter choices.
