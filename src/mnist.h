//
// Created by erenk on 8/6/2025.
//

#ifndef MNIST_H
#define MNIST_H
#include "linearalgebra.h"
#include "csv.h"

typedef struct {
    char digit;
    Vector pixels;
} MnistDigit;

MnistDigit* ReadDigitFromCSV(CSVFile *csv);
MnistDigit* CreateMnistDigit(char digit, Vector pixels);

#endif //MNIST_H