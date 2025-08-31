//
// Created by erenk on 7/24/2025.
//

#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H
#include "linearalgebra.h"

#define LEAK_COEFFICIENT 0.1

// List from https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions

double Identity(double x);
double BinaryStep(double x);
double LogisticSigmoid(double x);
double Tanh(double x);
double Relu(double x);
double LeakyRelu(double x);
//double ParametricLeakyRelu(double x, double a);

double IdentityPrime(double x);
double BinaryStepPrime(double x);
double LogisticSigmoidPrime(double x);
double TanhPrime(double x);
double ReluPrime(double x);
double LeakyReluPrime(double x);

Matrix* ActivateMatrix(Matrix *m, double (*aFunc)(double));

Vector* ActivateVector(Vector *v, double (*aFunc)(double));

#endif //ACTIVATIONFUNCTIONS_H
