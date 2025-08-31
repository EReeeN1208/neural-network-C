//
// Created by erenk on 8/6/2025.
//

#ifndef MNIST_H
#define MNIST_H
#include "linearalgebra.h"
#include "csv.h"
#include "nn.h"
#include "util.h"

#define MNIST_DIGIT_SIDE_LEN 28
#define MNIST_PIXEL_COUNT (MNIST_DIGIT_SIDE_LEN * MNIST_DIGIT_SIDE_LEN)

#define MNIST_DIGIT_COUNT 10

#define MNIST_LEARNING_RATE 0.001
#define MNIST_TRAINING_ROUNDS 3
#define MNIST_MAX_TRAINING_STEPS 30000
#define MNIST_TEST_TRAINING_STEP_INTERVAL 2000

typedef struct {
    char digit;
    Matrix* pixels;
} MnistDigit;

int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d);
MnistDigit* NewMnistDigit();

//these are mnist specific nNet functions so they are in mnist.h.
void TrainNetworkMNIST(NeuralNetwork *nNet, CSVFile *csvTrain, CSVFile *csvTest, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected));
TestResult* TestNetworkMNIST(NeuralNetwork  *nNet, CSVFile *csvTest, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected));
double CalculateLossMNIST(Tensor* probabilities, Tensor* outputGradient, int digit); //Precondition: output layer must be ran through softmax

TestResult* NewTestResultMNIST();

void FreeMnistDigit(MnistDigit *d);
void PrintMnistDigit(MnistDigit *d);

CSVFile* GetMNISTTestCSV();
CSVFile* GetMNISTTrainCSV();

#endif //MNIST_H