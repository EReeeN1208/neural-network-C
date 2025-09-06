# C Neural Network

Attempt to implement a neural network from scratch in C to enhance my Linear algebra, C and AI knowledge.



The current state of this project is fairly finished.
The project contains a modular system for creating Linear and Convolutional neural networks.
So far, only MNIST is implemented, but due to the systems modular nature, other classification datasets can also be easily implemented

However, there are aspects that could still be worked on.
I've noticed that larger convolutional networks can sometimes experience vanishing gradients.
There are also some other features that I believe could be implemented, such as GPU Support, a Dropout Layer and Network & Parameter saving/loading

## Building:

run `scripts/build_unix.sh` to build for linux and macos. the project will be built to `build/`and zipped to `out/`

## Training Data

Dataset: [Mnist](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

The data CSV Files were too large for GitHub. Extract the `mnist.zip` archive in `/data` into `/data`.

Note: the `prepare_data_unix.sh` is for builds. You will have to manually extract the csvs to run on the program for development. 

## Project Structure

The project is structured so that all general neural network functions lie in nn.h, and all mnist specific functions lie in mnist.h.

```
data
├── mnist.zip
├── mnist_test.csv --> unzip mnist.zip
└── mnist_train.csv --> unzip mnist.zip
scripts
├── build_unix.sh --> builds project for mac/linux
├── prepare_data_unix.sh --> copied into the build - not for use in/during development
└── run_unix.sh --> copied into the build - not for us in/during development
src
├── activationfunctions.h --> contains activation functions and their derivatives
├── csv.h --> contains code for opening, reading and parsing CSV files
├── linearalgebra.h --> contains most of the code for linear algebra & math (vector, matrix etc)
├── mnist.h --> contains all mnist specific neural network code
├── nn.h --> contains all neural network & network layer code (non-mnist specific)
├── tests.h --> contains some test functions for development and running/testing the neural network
├── util.h --> contains some util functions
├── main.c
└── corresponding .c files...
```

## To Do

- [ ] Add a batch normalization layer to fix vanishing gradients issue
- [ ] Add a dropout layer
- [ ] Implement Network & Parameter serialization & deserialization (saving/loading)
- [ ] Optimize Network Training/Execution (Perhaps cuda?)
- [ ] Better interface/ui/visualisation
- [ ] Refactor uType macros into enums (?)
