//
// Created by Eren Kural on 19.07.2025.
//


#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H


typedef struct {
    double* values;
    unsigned int size;
} Vector;

typedef struct {
    double* values;
    unsigned int r, c;
} Matrix;

Matrix* NewMatrix(unsigned int r, unsigned int c);
Matrix* InitMatrix(unsigned int r, unsigned int c, double fill);
Matrix* InitMatrixIncremental(unsigned int r, unsigned int c);
Matrix* GetIdentityMatrix(unsigned int len);
void FreeMatrix(Matrix *m);

Vector* NewVector(unsigned int size);
Vector* InitVector(unsigned int size, double fill);
Vector* InitVectorIncremental(unsigned int size);
void FreeVector(Vector *v);

Matrix* MatrixMultiply(Matrix *m1, Matrix *m2);
Vector* VectorMatrixMultiply(Matrix *m, Vector *v);

Matrix* MatrixAdd(Matrix *m1, Matrix *m2);
Vector* VectorAdd(Vector *v1, Vector *v2);

Matrix* TransposeMatrix(Matrix *m);

void ScaleMatrixDouble(Matrix *m, double s);
void ScaleVectorDouble(Vector *v, double s);
void ScaleMatrixInt(Matrix *m, int s);
void ScaleVectorInt(Vector *v, int s);

double GetMatrixValue(Matrix *m, unsigned int r, unsigned int c);
void SetMatrixValue(Matrix *m, unsigned int r, unsigned int c, double value);

void PrintMatrix(Matrix *m);
void PrintVector(Vector *v);

#endif //LINEARALGEBRA_H


