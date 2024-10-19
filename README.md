# Deep Dive into Neural Networks: Architecture and Training

This repository contains Python scripts demonstrating the architecture and training process of neural networks. It accompanies the Medium post "Deep Dive into Neural Networks: Architecture and Training".

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Visualizations](#visualizations)
4. [Topics Covered](#topics-covered)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run these scripts, you need Python 3.6 or later. Follow these steps to set up your environment:

1. Clone this repository:
   ```
   git clone https://github.com/ofrokon/neural-networks-deep-dive.git
   cd neural-networks-deep-dive
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate all visualizations and see the training process, run:

```
python neural_networks.py
```

This will create PNG files for visualizations and print training progress in the console.

## Visualizations

This script generates the following visualizations:

1. `activation_functions.png`: Plots of Sigmoid and ReLU activation functions
2. `training_loss.png`: Training loss over epochs for a simple neural network
3. `regularization_comparison.png`: Comparison of training loss with and without L2 regularization

## Topics Covered

1. Neural Network Architecture
2. Activation Functions
3. Feedforward Process
4. Backpropagation and Training
5. Loss Functions
6. Regularization (L2)
7. Simple implementation of a neural network from scratch

Each topic is explained in detail in the accompanying Medium post, including Python implementation and visualizations.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your suggested changes. If you're planning to make significant changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

---

For a detailed explanation of neural network concepts and their implementation, check out the accompanying Medium post: [Deep Dive into Neural Networks: Architecture and Training]([https://medium.com/yourusername/deep-dive-into-neural-networks](https://medium.com/@mroko001/deep-dive-into-neural-networks-architecture-and-training-b4258e4ab707))

For questions or feedback, please open an issue in this repository.
