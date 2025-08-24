//
// Created by erenk on 8/6/2025.
//

#ifndef MNIST_H
#define MNIST_H
#include "linearalgebra.h"
#include "csv.h"

#define MNIST_DIGIT_SIDE_LEN 28
#define MNIST_PIXEL_COUNT (MNIST_DIGIT_SIDE_LEN * MNIST_DIGIT_SIDE_LEN)

#define MNIST_DIGIT_COUNT 10

#define MNIST_LEARNING_RATE 0.01
#define MNIST_TRAINING_ROUNDS 4
#define MNIST_MAX_TRAINING_STEPS 30000

typedef struct {
    char digit;
    Matrix* pixels;
} MnistDigit;

int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d);
MnistDigit* NewMnistDigit();

double CalculateMnistLoss(Tensor* probabilities, Tensor* outputGradient, int digit); //Precondition: output layer must be ran through softmax

void FreeMnistDigit(MnistDigit *d);
void PrintMnistDigit(MnistDigit *d);

#endif //MNIST_H