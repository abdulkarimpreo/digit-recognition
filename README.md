# MNIST Digit Recognition

A handwritten digit recognition project built using Python and PyTorch.

## About the Project

This project uses a Neural Network to recognize handwritten digits (0–9) from the MNIST dataset.

The model is built using PyTorch and trained on 60,000 training images. It is evaluated using 10,000 test images.

## Model Architecture

The neural network contains:

- Input Layer: 784 neurons (28 × 28 pixels)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 64 neurons + ReLU
- Output Layer: 10 neurons (digits 0–9)

## Project Structure

```text
digit-recognition/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- MNIST Dataset

## Installation

Clone the repository:

```bash
git clone https://github.com/abdulkarimpreo/digit-recognition.git
```

Go to the project directory:

```bash
cd digit-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

The program will:

1. Load the MNIST dataset.
2. Create the neural network.
3. Train the model.
4. Evaluate model accuracy.
5. Save the trained model.
6. Predict a handwritten digit.

## Model Output

The trained model is saved in:

```text
models/mnist_model.pth
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.