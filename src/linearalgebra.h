//
// Created by Eren Kural on 19.07.2025.
//


#ifndef LINEARALGEBRA_H
#define LINEARALGEBRA_H

#define VECTOR 0
#define MATRIX2D 1
#define MATRIX3D 2

#define TENSOR_WRITE_MAX_BUFF 50

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
Vector* NewRandomisedVector(unsigned int size);
void FreeVector(Vector *v);

Matrix* NewEmptyMatrix(unsigned int r, unsigned int c);
Matrix* NewFilledMatrix(unsigned int r, unsigned int c, double fill);
Matrix* NewIncrementalMatrix(unsigned int r, unsigned int c);
Matrix* NewRandomisedMatrix(unsigned int r, unsigned int c);
Matrix* GetIdentityMatrix(unsigned int len);
void FreeMatrix(Matrix *m);

Matrix* NewXavierRandomMatrix(unsigned int r, unsigned int c);
Matrix* NewHeRandomMatrix(unsigned int r, unsigned int c);

Matrix3d* NewEmptyMatrix3d(unsigned int depth, unsigned int r, unsigned int c);
Matrix3d* NewFilledMatrix3d(unsigned int depth, unsigned int r, unsigned int c, double fill);
Matrix3d* NewRandomisedMatrix3d(unsigned int depth, unsigned int r, unsigned int c);
unsigned int GetMatrix3dSize(Matrix3d *m3d);
void FreeMatrix3d(Matrix3d *m);

Matrix* Get2dSliceMatrix3d(Matrix3d *m3d, unsigned int slice);
void Set2dSliceMatrix3d(Matrix3d *mDst, Matrix *mSrc, unsigned int slice);

Matrix* MatrixMultiply(Matrix *m1, Matrix *m2);
Vector* VectorMatrixMultiply(Matrix *m, Vector *v);
Matrix* NewMatrixSum(Matrix *m1, Matrix *m2);
Vector* NewVectorSum(Vector *v1, Vector *v2);
void AddVector(Vector *vDest, Vector *vToAdd); // Adds vToAdd to vDest. vDest is modified. vToAdd is not
void AddMatrix(Matrix *mDest, Matrix *mToAdd); // Adds vToAdd to vDest. vDest is modified. vToAdd is not
void AddDoubleToVector(Vector *vDest, double d);
void AddDoubleToMatrix(Matrix *mDest, double d);
Matrix* TransposeMatrix(Matrix *m);

void ScaleVectorDouble(Vector *v, double s);
void ScaleMatrixDouble(Matrix *m, double s);
void ScaleMatrix3dDouble(Matrix3d *m3d, double s);

double GetVectorValue(Vector *v, unsigned int pos);
void SetVectorValue(Vector *v, unsigned int pos, double value);
Vector* GetSubVector(Vector *v, unsigned int start, unsigned int size);

double GetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c);
double GetMatrixValuePos(Matrix *m, unsigned int pos);
void SetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c, double value);
void SetMatrixValuePos(Matrix *m, unsigned int pos, double value);

double GetMatrix3DValueDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c);
double GetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos);
void SetMatrix3DValueDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double value);
void SetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos, double value);
void AddDoubleToMatrix3dDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double valueToAdd);

Matrix* ConvolveMatrix(Matrix *image, Matrix *kernel);
Vector* FlattenMatrix(Matrix *m); // NOTE: Will reassign the pointer pointing the matrix's values to the vector and free the old matrix

Matrix* GetIdentityKernel(unsigned int size);
Matrix* GetEdgeDetectionKernel(); //3x3
Matrix* GetBlurKernel(unsigned int size);

Tensor* NewTensorEncapsulateVector(Vector* v);
Tensor* NewTensorEncapsulateMatrix(Matrix* m);
Tensor* NewTensorEncapsulateMatrix3d(Matrix3d* m3d);

Tensor* NewTensorCloneVector(Vector* v);
Tensor* NewTensorCloneMatrix(Matrix* m);
Tensor* NewTensorCloneMatrix3d(Matrix3d* m3d);

Tensor* CloneTensorEmpty(Tensor *t);
void CopyTensorValues(Tensor *tDst, Tensor *tSrc, _Bool checkTensorType); // Will check to make sure the sizes and types of tensors match
void ZeroTensorValues(Tensor *t);

double* GetTensorValues(Tensor *t);
double GetTensorValuePos(Tensor *t, unsigned int pos);
unsigned int GetTensorMaxIndex(Tensor *t); //return the index of the largest value inside a tensor's values
void FreeTensor(Tensor *t);

void AveragePoolMatrix(Matrix *mDst, Matrix *mSrc, unsigned int poolSize);
void MaxPoolMatrix(Matrix *mDst, Matrix *mSrc, unsigned int poolSize);
void AveragePoolMatrix3d(Matrix3d *mDst, Matrix3d *mSrc, unsigned int poolSize);
void MaxPoolMatrix3d(Matrix3d *mDst, Matrix3d *mSrc, unsigned int poolSize);

void PrintMatrix(Matrix *m);
void ShadeMatrix(Matrix *m);
void PrintVectorVertical(Vector *v);
void PrintVectorHorizontal(Vector *v);

void WriteTensorInfo(Tensor *t, char *buffer);

#endif //LINEARALGEBRA_H


