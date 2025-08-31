//
// Created by erenk on 7/24/2025.
//

#include "activationfunctions.h"

#include <math.h>

#include "linearalgebra.h"

double Identity(double x) {
    return x;
}

double BinaryStep(double x) {
    return x >= 0 ? 1 : 0;
}

double LogisticSigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double Tanh(double x) {
    //return (exp(x) - exp(-x)) / (exp(x) + exp (-x));
    return tanh(x);
}

double Relu(double x) {
    return x >= 0 ? x : 0;
}
double LeakyRelu(double x) {
    return x >= 0 ? x : LEAK_COEFFICIENT * x;
}
/*
double ParametricLeakyRelu(double x, double a) {
    return x >= 0 ? x : a * x;
}
*/

double IdentityPrime(double x) {
    return 1;
}
double BinaryStepPrime(double x) {
    return 0;
}
double LogisticSigmoidPrime(double x) {
    double a = LogisticSigmoid(x);
    return (1-a)*a;
}
double TanhPrime(double x) {
    return 1 - pow(Tanh(x), 2);
}
double ReluPrime(double x) {
    return x > 0 ? 1 : 0;
}
double LeakyReluPrime(double x) {
    return x > 0 ? 1 : LEAK_COEFFICIENT;
}

Matrix* ActivateMatrix(Matrix *m, double (*aFunc)(double)) {
    Matrix *mResult = NewEmptyMatrix(m->r, m->c);
    const unsigned int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        mResult->values[i] = aFunc(m->values[i]);
    }

    return mResult;
}

Vector* ActivateVector(Vector *v, double (*aFunc)(double)) {
    Vector *vResult = NewEmptyVector(v->size);

    for (int i = 0; i < v->size; i++) {
        vResult->values[i] = aFunc(v->values[i]);
    }
  
    return vResult;
}
