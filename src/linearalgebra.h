//
// Created by Eren Kural on 19.07.2025.
//


#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H


struct Vector {
    double* values;
    unsigned int size;
};

struct Matrix {
    double* values;
    unsigned int r, c;
};

struct Matrix NewMatrix(unsigned int r, unsigned int c);
struct Matrix InitMatrix(unsigned int r, unsigned int c, double fill);
struct Matrix InitMatrixIncremental(unsigned int r, unsigned int c);
struct Matrix GetIdentityMatrix(unsigned int len);
void DelMatrix(struct Matrix *m);

struct Vector NewVector(unsigned int size);
struct Vector InitVector(unsigned int size, double fill);
struct Vector InitVectorIncremental(unsigned int size);
void DelVector(struct Vector *v);

struct Matrix MatrixMultiply(struct Matrix *m1, struct Matrix *m2);
struct Vector VectorMatrixMultiply(struct Matrix *m, struct Vector *v);

struct Matrix MatrixAdd(struct Matrix *m1, struct Matrix *m2);
struct Vector VectorAdd(struct Vector *v1, struct Vector *v2);

struct Matrix TransposeMatrix(struct Matrix *m);

void ScaleMatrixDouble(struct Matrix *m, double s);
void ScaleVectorDouble(struct Vector *v, double s);
void ScaleMatrixInt(struct Matrix *m, int s);
void ScaleVectorInt(struct Vector *v, int s);

double GetMatrixValue(struct Matrix *m, unsigned int r, unsigned int c);
void SetMatrixValue(struct Matrix *m, unsigned int r, unsigned int c, double value);

void PrintMatrix(struct Matrix *m);
void PrintVector(struct Vector *v);

#endif //LINEARALGEBRA_H


