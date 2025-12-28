# MNIST Neural Network

A feed-forward neural network implementation for digit recognition using the MNIST dataset.

## Project Structure

```
mnist/
├── neural_network_mnist_numpy.py     # NumPy-optimized implementation
├── neural_network_mnist_for_loops.py # For-loop based implementation
├── data/                             # Data files
│   ├── mnist/                        # MNIST dataset files
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   └── t10k-labels-idx1-ubyte
│   ├── data_breast_cancer.p
│   └── neural_network.csv
├── weights/                          # Saved model weights
│   ├── weights_0.0001.npy
│   ├── weights_0.001.npy
│   └── weights.npy
├── requirements.txt                  # Python dependencies
└── README.md
```

## Installation

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the NumPy-optimized neural network:
```bash
python src/neural_network_mnist_numpy.py
```

Run the for-loop based implementation:
```bash
python src/neural_network_mnist_for_loops.py
```

## Configuration

The neural network can be configured by modifying parameters in the source files:
- `learning_rate` - Step size for gradient descent
- `epochs` - Number of training iterations
- `hidden_units` - Number of neurons in hidden layers
