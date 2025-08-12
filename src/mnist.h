//
// Created by erenk on 8/6/2025.
//

#ifndef MNIST_H
#define MNIST_H
#include "linearalgebra.h"
#include "csv.h"

#define MNIST_DIGIT_SIDE_LEN 28
#define MNIST_PIXEL_COUNT = MNIST_DIGIT_SIDE_LEN * MNIST_DIGIT_SIDE_LEN

typedef struct {
    char digit;
    Matrix* pixels;
} MnistDigit;

int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d);
MnistDigit* NewMnistDigit();

void FreeMnistDigit(MnistDigit *d);
void PrintMnistDigit(MnistDigit *d);

#endif //MNIST_H