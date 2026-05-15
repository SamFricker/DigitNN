# Digit NN — Handwritten Digit Recogniser

A handwritten digit classifier built from scratch in C++, trained on the MNIST dataset. No ML libraries, just raw neural network implementation with forward pass, backpropagation, and momentum-based gradient descent.

## Architecture

A fully connected feedforward neural network:

![Neural Network Architecture](images/architecture.png)

- **Input layer:** 784 nodes (28×28 pixel image)
- **Hidden layer 1:** 64 nodes, sigmoid activation
- **Hidden layer 2:** 16 nodes, sigmoid activation
- **Output layer:** 10 nodes (digits 0-9), sigmoid activation

## Features

- Xavier weight initialisation
- Sigmoid activation with overflow protection (clamped at ±50)
- SGD with momentum (η = 0.001, momentum = 0.9)
- Gradient clipping (threshold = 1.0)
- Mini-batch training support
- Parameter save/load to disk
- Custom image preprocessing pipeline for user-drawn digits:
  - Grayscale loading via `stb_image`
  - Centre-of-mass alignment
  - Moment-based deskewing
  - Morphological closing (3×3 dilate + erode)

## Building

Using MinGW/g++ (MSYS2):

## Running

fix.cpp press **F5** in VS Code.

> **Note:** Running by typing the path directly in PowerShell without `&` will not execute the binary due to a PowerShell quirk with paths containing spaces.

## Project Structure

```
Digit NN/
├── src/
│   ├── fix.cpp          # Main source file
│   └── stb_image.h      # Image loading library
├── data/
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   └── Test_Digit=*.png # test images
├── parameters/
│   └── perameters.txt   # Saved network weights
└── .vscode/
    ├── tasks.json
    └── launch.json
```

## Results

Trained on the MNIST test set (10,000 images):

| Training method | Epochs | Accuracy |
|---|---|---|
| SGD (one image at a time) | 3 | ~93% |
| Mini-batch SGD (batch=32) | 2 | ~93.5% |

> Training on the full 60,000-image training set would likely push accuracy above 96–97%.

## Testing on Your Own Digits

Place a PNG image of a handwritten digit in the `data/` folder and call:

```cpp
net.test("path\\to\\your\\image.png");
```

The image can be any size: it will be rescaled, deskewed, and normalised to 28×28 automatically.

## Next Steps and Reflections
Overall the project was very sucessful, with minimal trainign the NN has acheived up to 95% accuracy on handwritten digits and has features which allow it to be easily applied, the project taught me the raw mathematics behind neural networks and i would like to now explore its applications especially in robotics. This would be quite easy to implement given enough training data since we could just change the MNIST data set for any other data set formatted in the same way and retrain the model, to give it a new purpose. Otherwise, full comments are still needed to properly document the process.
