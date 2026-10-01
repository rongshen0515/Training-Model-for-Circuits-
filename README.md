# MNIST for Analog Circuits

This project trains a simple Convolutional Neural Network (CNN) to classify hand-drawn electrical circuit components.

The dataset is based on the JUHCCR-v1 circuit component recognition dataset.

Eventually I will develop a stronger model to read schematics instead of just symbols.

## Classes Used

This project uses 6 circuit component classes:

- Capacitor
- Diode
- Inductor
- NPN transistor
- PNP transistor
- Resistor

## Project Structure

The project folder should look like this:

```text
circuit_project/
│
├── train_cnn.py
│
├── dataset/
│   ├── Capacitor/
│   ├── Diode/
│   ├── Inductor/
│   ├── NPN/
│   ├── PNP/
│   └── Resistor/
│
└── README.md
