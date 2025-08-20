//
// Created by erenk on 8/6/2025.
//

#include "mnist.h"

#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

#include "util.h"


int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d) {

    char lineBuffer[CSV_LINE_MAX_BUFF];
    char valueBuffer[CSV_VALUE_MAX_BUFF];

    int lineNum = GetNextLine(csv, lineBuffer);

    if (lineNum == -1) {
        return -1;
    }

    Vector *vLine = NewEmptyVector(1 + MNIST_PIXEL_COUNT);

    unsigned int lineLen = strlen(lineBuffer);
    unsigned int i = 0; // lineBuffer iterator;
    unsigned int j = 0; // valueBuffer iterator
    unsigned int k = 0; // vLine iterator

    while (i<lineLen && lineBuffer[i] != '\0') {
        valueBuffer[j++] = lineBuffer[i++];
        if (lineBuffer[i] == ',') {
            i++;
            valueBuffer[j] = '\0';
            j = 0;
            SetVectorValue(vLine, k++, StrToUChar(valueBuffer, CSV_VALUE_MAX_BUFF));
        }
    }

    d->digit = (char)GetVectorValue(vLine, 0);

    for (int px = 0; px<MNIST_PIXEL_COUNT; px++) {
        SetMatrixValuePos(d->pixels, px, GetVectorValue(vLine, px+1) / 255.0);
    }

    FreeVector(vLine);
    return 0;
}

MnistDigit* NewMnistDigit() {
    MnistDigit *d = malloc(sizeof(MnistDigit));
    d->pixels = NewEmptyMatrix(MNIST_DIGIT_SIDE_LEN, MNIST_DIGIT_SIDE_LEN);
    return d;
}

double CalculateMnistLoss(Tensor* probabilities, Tensor* outputGradient, int digit) {
    if (probabilities->size != MNIST_DIGIT_COUNT || outputGradient->size != MNIST_DIGIT_COUNT) {
        fprintf(stderr, "Error: tried to calculate MNIST loss for tensor with invalid sizes: %d, %d", probabilities->size, outputGradient->size);
        exit(EXIT_FAILURE_CODE);
    }
    // One possible function. Apparently, its not great
    /*
    return 1 - pow(GetTensorValuePos(probabilities, digit), 2);
    */


    // Mean Squared Error
    /*
    double totalError = 0;

    for (int i = 0; i<10; i++) {
        if (i != digit) {
            totalError += 0.5 * pow(0 - GetTensorValuePos(probabilities, i), 2);
        }
    }
    totalError += 0.5 * pow(1 - GetTensorValuePos(probabilities, digit), 2);

    return totalError;
    */


    // Cross Entropy (Apparently this is the best method for classification tasks like mnist)

    double oneHotVector[MNIST_DIGIT_COUNT] = { 0 };
    oneHotVector[digit] = 1;

    const double epsilon = 1e-15;
    double prob = GetTensorValuePos(probabilities, digit);
    prob = fmax(prob, epsilon);  // Ensure prob >= epsilon

    for (int i = 0; i < MNIST_DIGIT_COUNT; i++) {
        double gradient = GetTensorValues(probabilities)[i] - oneHotVector[i];
        // clamp gradients to prevent explosion
        gradient = fmax(-10.0, fmin(10.0, gradient));
        GetTensorValues(outputGradient)[i] = gradient;
    }

    return -1.0 * log(prob); //return loss value. not really necessary
}

void FreeMnistDigit(MnistDigit *d) {
    if (d->pixels != NULL) {
        FreeMatrix(d->pixels);
    }
    free(d);
}

void PrintMnistDigit(MnistDigit *d) {
    ShadeMatrix(d->pixels);
}
