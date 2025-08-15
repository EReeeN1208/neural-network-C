//
// Created by Eren Kural on 19.07.2025.
//


#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#define VECTOR 0
#define MATRIX2D 1
#define MATRIX3D 2

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

typedef struct {
    double* values;
    unsigned int depth, r, c;
} Matrix3d;

typedef struct {
    unsigned int uType;
    unsigned int size;
    union {
        Vector *vector;
        Matrix *matrix2d;
        Matrix3d *matrix3d;
    };
} Tensor;

Vector* NewEmptyVector(unsigned int size);
Vector* NewFilledVector(unsigned int size, double fill);
Vector* NewIncrementalVector(unsigned int size);
void FreeVector(Vector *v);

Matrix* NewEmptyMatrix(unsigned int r, unsigned int c);
Matrix* NewFilledMatrix(unsigned int r, unsigned int c, double fill);
Matrix* NewIncrementalMatrix(unsigned int r, unsigned int c);
Matrix* NewRandomisedMatrix(unsigned int r, unsigned int c);
Matrix* GetIdentityMatrix(unsigned int len);
void FreeMatrix(Matrix *m);

Matrix3d* NewEmptyMatrix3d(unsigned int depth, unsigned int r, unsigned int c);
Matrix3d* NewFilledMatrix3d(unsigned int depth, unsigned int r, unsigned int c, double fill);
Matrix3d* NewRandomisedMatrix3d(unsigned int depth, unsigned int r, unsigned int c);
unsigned int GetMatrix3dSize(Matrix3d *m3d);
void FreeMatrix3d(Matrix3d *m);

Matrix* MatrixMultiply(Matrix *m1, Matrix *m2);
Vector* VectorMatrixMultiply(Matrix *m, Vector *v);
Matrix* MatrixAdd(Matrix *m1, Matrix *m2);
Vector* VectorAdd(Vector *v1, Vector *v2);
Matrix* TransposeMatrix(Matrix *m);

void ScaleVectorDouble(Vector *v, double s);
void ScaleVectorInt(Vector *v, int s);
void ScaleMatrixDouble(Matrix *m, double s);
void ScaleMatrixInt(Matrix *m, int s);
void ScaleMatrix3dDouble(Matrix3d *m3d, double s);
void ScaleMatrix3dInt(Matrix3d *m3d, int s);

double GetVectorValue(Vector *v, unsigned int pos);
void SetVectorValue(Vector *v, unsigned int pos, double value);
Vector* GetSubVector(Vector *v, unsigned int start, unsigned int size);

double GetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c);
double GetMatrixValuePos(Matrix *m, unsigned int pos);
void SetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c, double value);
void SetMatrixValuePos(Matrix *m, unsigned int pos, double value);

double GetMatrix3DValueRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c);
double GetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos);
void SetMatrix3DValueRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double value);
void SetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos, double value);

Matrix* ConvolveMatrix(Matrix *image, Matrix *kernel);
Vector* FlattenMatrix(Matrix *m); // NOTE: Will reassign the pointer pointing the matrix's values to the vector and free the old matrix

Matrix* GetIdentityKernel(unsigned int size);
Matrix* GetEdgeDetectionKernel(); //3x3
Matrix* GetBlurKernel(unsigned int size);

Tensor* NewTensorVector(Vector* v);
Tensor* NewTensorMatrix(Matrix* m);
Tensor* NewTensorMatrix3d(Matrix3d* m3d);
Tensor* CloneTensorEmpty(Tensor *t);

double* GetTensorValues(Tensor *t);
double GetTensorValuePos(Tensor *t, unsigned int pos);
void FreeTensor(Tensor *t);


void PrintMatrix(Matrix *m);
void ShadeMatrix(Matrix *m);
void PrintVectorVertical(Vector *v);
void PrintVectorHorizontal(Vector *v);

#endif //LINEARALGEBRA_H


