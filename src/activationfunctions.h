//
// Created by erenk on 7/24/2025.
//

#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include "linearalgebra.h"

// List from https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions

double Identity(double x);
double BinaryStep(double x);
double LogisticSigmoid(double x);
double Tanh(double x);
double Relu(double x);
double LeakyRelu(double x);
//double ParametricLeakyRelu(double x, double a);

Matrix* ActivateMatrix(Matrix *m, double (*aFunc)(double));

Vector* ActivateVector(Vector *v, double (*aFunc)(double));

#endif //ACTIVATIONFUNCTIONS_H
