//
// Created by Eren Kural on 19.07.2025.
//


#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

/*
typedef union {
    double dVal; // 0
    long lVal; // 1
} Value;

typedef struct {
    Value* values;
    char vType;
    unsigned int size;
} Vector;

typedef struct {
    Value* values;
    char vType;
    unsigned int r, c;
} Matrix;
*/

typedef struct {
    double* values;
    unsigned int size;
} Vector;

typedef struct {
    double* values;
    unsigned int r, c;
} Matrix;

Matrix* NewEmptyMatrix(unsigned int r, unsigned int c);
Matrix* NewFilledMatrix(unsigned int r, unsigned int c, double fill);
Matrix* NewIncrementalMatrix(unsigned int r, unsigned int c);
Matrix* GetIdentityMatrix(unsigned int len);
void FreeMatrix(Matrix *m);

Vector* NewEmptyVector(unsigned int size);
Vector* NewFilledVector(unsigned int size, double fill);
Vector* NewIncrementalVector(unsigned int size);
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

double GetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c);
double GetMatrixValuePos(Matrix *m, unsigned int pos);
void SetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c, double value);
void SetMatrixValuePos(Matrix *m, unsigned int pos, double value);

double GetVectorValue(Vector *v, unsigned int pos);
void SetVectorValue(Vector *v, unsigned int pos, double value);
Vector* GetSubVector(Vector *v, unsigned int start, unsigned int size);

Matrix* ConvolveMatrix(Matrix *image, Matrix *kernel);
Vector* FlattenMatrix(Matrix *m); // NOTE: Will free previous matrix and assign the pointer to its values to the vectors values

Matrix* GetIdentityKernel(unsigned int size);
Matrix* GetEdgeDetectionKernel(); //3x3
Matrix* GetBlurKernel(unsigned int size);

void PrintMatrix(Matrix *m);
void ShadeMatrix(Matrix *m);
void PrintVector(Vector *v);

#endif //LINEARALGEBRA_H


