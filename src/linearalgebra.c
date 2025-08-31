//
// Created by Eren Kural on 19.07.2025.
//
#include "linearalgebra.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

#include "util.h"


Vector* NewEmptyVector(unsigned int size) {
    Vector *v = malloc(sizeof(Vector));
    v->size = size;
    v->values = calloc(size, sizeof(double));

    return v;
}
Vector* NewFilledVector(unsigned int size, double fill) {
    Vector *v = NewEmptyVector(size);

    for (int i = 0; i < size; i++) {
        v->values[i] = fill;
    }

    return v;
}

Vector* NewIncrementalVector(unsigned int size) {
    Vector *v = NewEmptyVector(size);

    for (int i = 0; i < size; i++) {
        v->values[i] = i;
    }
    return v;
}

Vector* NewRandomisedVector(unsigned int size) {
    Vector *v = NewEmptyVector(size);

    for (int i = 0; i < size; i++) {
        v->values[i] = GetRandomNormalised();
    }
    return v;
}


void FreeVector(Vector *v) {
    if (v == NULL) {
        return;
    }
    if (v->values != NULL) {
        free(v->values);
    }
    free(v);
}



Matrix* NewEmptyMatrix(unsigned int r, unsigned int c) {
    Matrix *m = malloc(sizeof(Matrix));
    const unsigned int size = r * c;
    m->r = r;
    m->c = c;
    m->values = calloc(size, sizeof(double));

    return m;
}

Matrix* NewFilledMatrix(unsigned int r, unsigned int c, double fill) {
    Matrix *m = NewEmptyMatrix(r, c);

    const unsigned int size = r * c;

    for (int i = 0; i < size; i++) {
        m->values[i] = fill;
    }


    return m;
}

Matrix* NewIncrementalMatrix(unsigned int r, unsigned int c) {
    Matrix *m = NewEmptyMatrix(r, c);

    const unsigned int size = r * c;

    for (int i = 0; i < size; i++) {
        m->values[i] = i;
    }
    return m;
}

Matrix* NewRandomisedMatrix(unsigned int r, unsigned int c) {
    Matrix *m = NewEmptyMatrix(r, c);

    const unsigned int size = r * c;

    for (int i = 0; i < size; i++) {
        m->values[i] = GetRandomNormalised();
    }


    return m;
}

Matrix* GetIdentityMatrix(unsigned int len) {
    Matrix *m = NewEmptyMatrix(len, len);
    for (unsigned i = 0; i < len*len; i += (len + 1)) {
        m->values[i] = 1;
    }
    return m;
}

void FreeMatrix(Matrix *m) {
    free(m->values);
    free(m);
}


Matrix* NewXavierRandomMatrix(unsigned int r, unsigned int c) {
    Matrix *m = NewEmptyMatrix(r, c);

    const unsigned int size = r * c;
    double xavier_scale = sqrt(6.0 / (double)(r + c));

    for (int i = 0; i < size; i++) {
        m->values[i] = GetRandomNormalised() * xavier_scale;
    }

    return m;
}


Matrix* NewHeRandomMatrix(unsigned int r, unsigned int c) {
    Matrix *m = NewEmptyMatrix(r, c);

    const unsigned int size = r * c;

    double he_scale = sqrt(2.0 / (double)c);

    for (int i = 0; i < size; i++) {
        m->values[i] = GetRandomNormalised() * he_scale;
    }

    return m;
}



Matrix3d* NewEmptyMatrix3d(unsigned int depth, unsigned int r, unsigned int c) {
    Matrix3d *m = malloc(sizeof(Matrix3d));
    const unsigned int size = depth * r * c;
    m->depth = depth;
    m->r = r;
    m->c = c;

    m->values = calloc(size, sizeof(double));

    return m;
}

Matrix3d* NewFilledMatrix3d(unsigned int depth, unsigned int r, unsigned int c, double fill) {
    Matrix3d *m = NewEmptyMatrix3d(depth, r, c);

    const unsigned int size = depth * r * c;

    for (int i = 0; i < size; i++) {
        m->values[i] = fill;
    }

    return m;
}

Matrix3d* NewRandomisedMatrix3d(unsigned int depth, unsigned int r, unsigned int c) {
    Matrix3d *m = NewEmptyMatrix3d(depth, r, c);

    const unsigned int size = depth * r * c;

    for (int i = 0; i < size; i++) {
        m->values[i] = GetRandomNormalised();
    }

    return m;
}

unsigned int GetMatrix3dSize(Matrix3d *m3d) {
    return m3d->depth * m3d->r * m3d->c;
}

void FreeMatrix3d(Matrix3d *m) {
    free(m->values);
    free(m);
}



Matrix* Get2dSliceMatrix3d(Matrix3d *m3d, unsigned int slice) {
    if (slice >= m3d->depth) {
        fprintf(stderr, "Attempted to access slice out of bounds. Get2dSliceMatrix3d()");
        exit(EXIT_FAILURE_CODE);
    }

    unsigned int sliceSize = m3d->r * m3d->c;

    Matrix *m = NewEmptyMatrix(m3d->r, m3d->c);
    memcpy(m->values, m3d->values+sliceSize*slice, sizeof(double) * sliceSize);

    return m;
}

void Set2dSliceMatrix3d(Matrix3d *mDst, Matrix *mSrc, unsigned int slice) {
    if (slice >= mDst->depth) {
        fprintf(stderr, "Attempted to access slice out of bounds. %d, %d. Set2dSliceMatrix3d()", slice, mDst->depth);
        exit(EXIT_FAILURE_CODE);
    }
    unsigned int sliceSize = mDst->r * mDst->c;

    memcpy(mDst->values+sliceSize*slice, mSrc->values, sizeof(double) * sliceSize);
}



Matrix* MatrixMultiply(Matrix *m1, Matrix *m2) {
    if (m1->c != m2->r) {
        fprintf(stderr, "Error during matrix matrix multiplication. Sizes: %dx%d and %dx%d", m1->r, m1->c, m2->r, m2->c);
        exit(EXIT_FAILURE_CODE);
    }
    int size = m1->r * m2->c;
    Matrix *mResult = NewEmptyMatrix(m1->r, m2->c);

    double sum;
    int r, c;
    for (int i = 0; i < size; i++) {
        r = i / mResult->c;
        c = i % mResult->c;
        sum = 0;

        for (int j = 0; j < m1->c; j++) {
            sum += GetMatrixValueRowCol(m1, r, j) * GetMatrixValueRowCol(m2, j, c);
        }
        mResult->values[i] = sum;
    }

    return mResult;
}

Vector* VectorMatrixMultiply(Matrix *m, Vector *v) {
    if (m->c != v->size) {
        fprintf(stderr, "Error during matrix vector multiplication. Sizes: %dx%d and v%d", m->r, m->c, v->size);
        exit(EXIT_FAILURE_CODE);
    }

    Vector *vResult = NewEmptyVector(m->r);

    double sum;
    for (int i = 0; i < m->r; i++) {
        sum = 0;
        for (int j = 0; j < m->c; j++) {
            sum += GetMatrixValueRowCol(m, i, j)*v->values[j];
        }
        vResult->values[i] = sum;
    }
    return vResult;
}

Matrix* NewMatrixSum(Matrix *m1, Matrix *m2) {
    if (m1->r != m2->r || m1->c != m2->c) {
        fprintf(stderr, "Matrices not same size for addition. NewMatrixSum()");
        exit(EXIT_FAILURE_CODE);
    }
    unsigned int size = m1->r * m1->c;

    Matrix *mResult = NewEmptyMatrix(m1->r, m1->c);

    for (int i = 0; i < size; i++) {
        mResult->values[i] = m1->values[i] + m2->values[i];
    }
    return mResult;
}

Vector* NewVectorSum(Vector *v1, Vector *v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Vectors not same size for addition. NewVectorSum()");
        exit(EXIT_FAILURE_CODE);
    }

    Vector *vResult = NewEmptyVector(v1->size);

    for (int i = 0; i < v1->size; i++) {
        vResult->values[i] = v1->values[i] + v2->values[i];
    }
    return vResult;
}

void AddVector(Vector *vDest/*changes*/, Vector *vToAdd/*remains unchanged*/) {
    if (vDest->size != vToAdd->size) {
        fprintf(stderr, "Vectors not same size for addition. AddVector()");
        exit(EXIT_FAILURE_CODE);
    }

    for (int i = 0; i < vDest->size; i++) {
        vDest->values[i] += vToAdd->values[i];
    }

}

void AddMatrix(Matrix *mDest, Matrix *mToAdd) {
    if (mDest->r != mToAdd->r || mDest->c != mToAdd->c) {
        fprintf(stderr, "Matrices not same size for addition. AddMatrix()");
        exit(EXIT_FAILURE_CODE);
    }

    unsigned int size = mDest->r * mDest->c;

    for (int i = 0; i<size; i++) {
        mDest->values[i] += mToAdd->values[i];
    }
}

void AddDoubleToVector(Vector *vDest, double d) {
    for (int i = 0; i < vDest->size; i++) {
        vDest->values[i] += d;
    }
}
void AddDoubleToMatrix(Matrix *mDest, double d) {
    unsigned int size = mDest->r * mDest->c;

    for (int i = 0; i<size; i++) {
        mDest->values[i] += d;
    }
}

Matrix* TransposeMatrix(Matrix *m) {
    unsigned int size = m->r * m->c;

    Matrix* mResult = NewEmptyMatrix(m->c, m->r);
    /*
    for (int i = 0; i < size; i++) {
        mResult->values[i] = GetMatrixValueRowCol(m, i % m->c, i / m->c);
    }
    */
    for (int i = 0; i<m->r; i++) {
        for (int j = 0; j<m->c; j++) {
            SetMatrixValueRowCol(mResult, j, i, GetMatrixValueRowCol(m, i, j));
        }
    }
    return mResult;
}


void ScaleVectorDouble(Vector *v, double s) {
    for (int i = 0; i < v->size; i++) {
        v->values[i] *= s;
    }
}
void ScaleMatrixDouble(Matrix *m, double s) {
    unsigned int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        m->values[i] *= s;
    }
}
void ScaleMatrix3dDouble(Matrix3d *m3d, double s) {
    unsigned int size = m3d->depth * m3d->r * m3d->c;

    for (int i = 0; i < size; i++) {
        m3d->values[i] *= s;
    }
}



double GetVectorValue(Vector *v, unsigned int pos) {
    if (pos>=v->size) {
        fprintf(stderr, "Error during vector value access. Vector size: %d, Tried to access: %d", v->size, pos);
        exit(EXIT_FAILURE_CODE);
    }
    return v->values[pos];
}
void SetVectorValue(Vector *v, unsigned int pos, double value) {
    if (pos>=v->size) {
        fprintf(stderr, "Error during vector value set. Vector size: %d, Tried to set: %d", v->size, pos);
        exit(EXIT_FAILURE_CODE);
    }
    v->values[pos] = value;
}

/* Start counting from 0 */
Vector* GetSubVector(Vector *v, unsigned int start, unsigned int size) {
    if ((start + size)>v->size) {
        fprintf(stderr, "Error during vector sub-vector access. Vector size: %d, Tried to access %d values starting from %d", v->size, size, start);
        exit(EXIT_FAILURE_CODE);
    }

    Vector *nResult = NewEmptyVector(size);

    for (int i = 0; i<size; i++) {
        SetVectorValue(nResult, i, GetVectorValue(v, start + i));
    }
    return nResult;
}



/* Start counting from 0 */
double GetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c) {
    if (r >= m->r || c >= m->c) {
        fprintf(stderr, "Error during matrix value get w/ r-c. Matrix size: %dx%d, Tried to access: %dx%d", m->r, m->c, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    return m->values[r*m->c + c];
}

double GetMatrixValuePos(Matrix *m, unsigned int pos) {
    if (pos >= m->c * m->r) {
        fprintf(stderr, "Error during matrix value get w/ pos. Matrix size: %dx%d, Tried to get pos: %d", m->r, m->c, pos);
        exit(EXIT_FAILURE_CODE);
    }
    return m->values[pos];
}

/* Start counting from 0 */
void SetMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c, double value) {
    if (r >= m->r || c >= m->c) {
        fprintf(stderr, "Error during matrix value set w/ r-c. Matrix size: %dx%d, Tried to set: %dx%d", m->r, m->c, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    m->values[r*m->c + c] = value;
}

void SetMatrixValuePos(Matrix *m, unsigned int pos, double value) {
    if (pos >= m->c * m->r) {
        fprintf(stderr, "Error during matrix value set w/ pos. Matrix size: %dx%d, Tried to set pos: %d", m->r, m->c, pos);
        exit(EXIT_FAILURE_CODE);
    }
    m->values[pos] = value;
}

void IncrementMatrixValueRowCol(Matrix *m, unsigned int r, unsigned int c, double increment) {
    if (r >= m->r || c >= m->c) {
        fprintf(stderr, "Error during matrix value set w/ r-c. Matrix size: %dx%d, Tried to set: %dx%d", m->r, m->c, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    m->values[r*m->c + c] += increment;
}



double GetMatrix3DValueDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c) {
    if (depth >= m3d->depth || r >= m3d->r || c >= m3d->c) {
        fprintf(stderr, "Error during matrix3d value get w/ d-r-c. Matrix size: %dx%dx%d, Tried to get: %dx%dx%d", m3d->depth, m3d->r, m3d->c, depth, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    return m3d->values[depth*m3d->r*m3d->c + r*m3d->c + c];
}

double GetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos) {
    if (pos >= GetMatrix3dSize(m3d)) {
        fprintf(stderr, "Error during matrix3d value get w/ pos. Matrix size: %dx%dx%d, Tried to get pos: %d", m3d->depth, m3d->r, m3d->c, pos);
        exit(EXIT_FAILURE_CODE);
    }
    return m3d->values[pos];
}

void SetMatrix3DValueDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double value) {
    if (depth >= m3d->depth || r >= m3d->r || c >= m3d->c) {
        fprintf(stderr, "Error during matrix3d value set w/ d-r-c. Matrix size: %dx%dx%d, Tried to set: %dx%dx%d", m3d->depth, m3d->r, m3d->c, depth, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    m3d->values[depth*m3d->r*m3d->c + r*m3d->c + c] = value;
}

void SetMatrix3DValuePos(Matrix3d *m3d, unsigned int pos, double value) {
    if (pos >= GetMatrix3dSize(m3d)) {
        fprintf(stderr, "Error during matrix3d value set w/ pos. Matrix size: %dx%dx%d, Tried to set pos: %d", m3d->depth, m3d->r, m3d->c, pos);
        exit(EXIT_FAILURE_CODE);
    }
    m3d->values[pos] = value;
}

void AddDoubleToMatrix3dDepthRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double valueToAdd) {
    if (depth >= m3d->depth || r >= m3d->r || c >= m3d->c) {
        fprintf(stderr, "Error during AddDoubleToMatrix3dDepthRowCol Matrix size: %dx%dx%d, Tried to set: %dx%dx%d", m3d->depth, m3d->r, m3d->c, depth, r, c);
        exit(EXIT_FAILURE_CODE);
    }
    m3d->values[depth*m3d->r*m3d->c + r*m3d->c + c] += valueToAdd;
}



Matrix* ConvolveMatrix(Matrix *image, Matrix *kernel) {
    if (kernel->r != kernel->c || kernel->r % 2 != 1) {
        fprintf(stderr, "Error during image convolution with filter size: %dx%d", kernel->r, kernel->c);
        exit(EXIT_FAILURE_CODE);
    }
    unsigned int ks = kernel->r; // kernel size
    unsigned int kr = ks / 2; //kernelRadius
    unsigned int mR = image->r - 2*kr;
    unsigned int mC = image->c - 2*kr;


    Matrix *m = NewEmptyMatrix(mR, mC);

    //i and j iterate starting from 0 to mR and mC (the size of matrix m).
    for (unsigned int i = 0; i<mR; i++) {
        for (unsigned int j = 0; j<mC; j++) {
            double value = 0;
            for (int k = 0; k<(ks*ks); k++) {
                value += GetMatrixValuePos(kernel, k) * GetMatrixValueRowCol(image, i + (k/ks), j + (k%ks));
            }
            SetMatrixValueRowCol(m, i, j, value);
        }
    }

    return m;
}

Vector* FlattenMatrix(Matrix *m) {
    Vector *v = malloc(sizeof(Vector));
    v->size = m->r * m->c;
    v->values = m->values;
    free(m);
    return v;
}



Matrix* GetIdentityKernel(unsigned int size) {
    if (size % 2 != 1) {
        fprintf(stderr, "Invalid convolution identity matrix size %d", size);
        exit(EXIT_FAILURE_CODE);
    }
    Matrix *m = NewEmptyMatrix(size, size);
    SetMatrixValueRowCol(m, size/2, size/2, 1);
    return m;
}

Matrix* GetEdgeDetectionKernel() {
    Matrix *m = NewFilledMatrix(3, 3, -1);
    SetMatrixValueRowCol(m, 1, 1, 8);
    return m;
}

Matrix* GetBlurKernel(unsigned int size) {
    if (size % 2 != 1) {
        fprintf(stderr, "Invalid convolution identity matrix size %d", size);
        exit(EXIT_FAILURE_CODE);
    }
    Matrix *m = NewFilledMatrix(size, size, 1.0/(size * size));
    return m;
}



Tensor* NewTensorEncapsulateVector(Vector* v) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = VECTOR;
    t->vector = v;
    t->size = v->size;

    return t;
}
Tensor* NewTensorEncapsulateMatrix(Matrix* m) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX2D;
    t->matrix2d = m;
    t->size = m->r * m->c;

    return t;
}
Tensor* NewTensorEncapsulateMatrix3d(Matrix3d* m3d) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX3D;
    t->matrix3d = m3d;
    t->size = GetMatrix3dSize(m3d);

    return t;
}

Tensor* NewTensorCloneVector(Vector* v) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = VECTOR;
    t->vector = NewEmptyVector(v->size);
    t->size = v->size;

    memcpy(t->vector->values, v->values, sizeof(double) * t->size);

    return t;
}
Tensor* NewTensorCloneMatrix(Matrix* m) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX2D;
    t->matrix2d = NewEmptyMatrix(m->r, m->c);
    t->size = m->r * m->c;

    memcpy(t->matrix2d->values, m->values, sizeof(double) * t->size);

    return t;
}
Tensor* NewTensorCloneMatrix3d(Matrix3d* m3d) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX3D;
    t->matrix3d = NewEmptyMatrix3d(m3d->depth, m3d->r, m3d->c);
    t->size = GetMatrix3dSize(m3d);

    memcpy(t->matrix3d->values, m3d->values, sizeof(double) * t->size);

    return t;
}

Tensor* CloneTensorEmpty(Tensor *t) {
    Tensor *tNew = calloc(1, sizeof(Tensor));
    tNew->size = t->size;
    tNew->uType = t->uType;

    switch (t->uType) {
        case VECTOR: {
            tNew->vector = NewEmptyVector(t->vector->size);
            break;
        }
        case MATRIX2D: {
            tNew->matrix2d = NewEmptyMatrix(t->matrix2d->r, t->matrix2d->c);
            break;
        }
        case MATRIX3D: {
            tNew->matrix3d = NewEmptyMatrix3d(t->matrix3d->depth, t->matrix3d->r, t->matrix3d->c);
            break;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor CloneTensorEmpty().", t->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }

    return tNew; // does not leak even tough static analysis says it does
}

void CopyTensorValues(Tensor *tDst, Tensor *tSrc, _Bool checkTensorType) {
    if ((tDst->uType != tSrc->uType) && checkTensorType) {
        fprintf(stderr, "Error: The input and output tensor's are not compatible. CopyTensorValues()");
        exit(EXIT_FAILURE_CODE);
    }
    if (tDst->size != tSrc->size) {
        fprintf(stderr, "Error: The input and output tensors' sizes do not match. %d, %d. CopyTensorValues()", tDst->size, tSrc->size);
        exit(EXIT_FAILURE_CODE);
    }

    memcpy(GetTensorValues(tDst), GetTensorValues(tSrc), sizeof(double) * tSrc->size);
}

void ZeroTensorValues(Tensor *t) {
    switch (t->uType) {
        case VECTOR: {
            free(t->vector->values);
            t->vector->values = calloc(t->size, sizeof(double));
            break;
        }
        case MATRIX2D: {
            free(t->matrix2d->values);
            t->matrix2d->values = calloc(t->size, sizeof(double));
            break;
        }
        case MATRIX3D: {
            free(t->matrix3d->values);
            t->matrix3d->values = calloc(t->size, sizeof(double));
            break;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor ZeroTensorValues().", t->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
}

void ZeroValues(double *values, unsigned int size) {
    for (int i = 0; i<size; i++) {
        values[i] = 0;
    }
}

double* GetTensorValues(Tensor *t) {
    switch (t->uType) {
        case VECTOR: {
            return t->vector->values;
        }
        case MATRIX2D: {
            return t->matrix2d->values;
        }
        case MATRIX3D: {
            return t->matrix3d->values;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor GetTensorValues().", t->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
}

double GetTensorValuePos(Tensor *t, unsigned int pos) {
    if (pos >= t->size) {
        fprintf(stderr, "Error: tried to access out of bounds pos '%d' for a tensor. GetTensorValuePos().", pos);
        exit(EXIT_FAILURE_CODE);
    }
    return GetTensorValues(t)[pos];
}

unsigned int GetTensorMaxIndex(Tensor *t) {
    const double *tensorValues = GetTensorValues(t);

    unsigned int index = 0;
    double max = tensorValues[0];

    for (unsigned int i = 1; i<t->size; i++) {
        if (tensorValues[i]>max) {
            max = tensorValues[i];
            index = i;
        }
    }
    return index;
}

unsigned int GetTensorMinIndex(Tensor *t) {
    const double *tensorValues = GetTensorValues(t);

    unsigned int index = 0;
    double min = tensorValues[0];

    for (unsigned int i = 1; i<t->size; i++) {
        if (tensorValues[i]<min) {
            min = tensorValues[i];
            index = i;
        }
    }
    return index;
}

void FreeTensor(Tensor *t) {
    //printf("\n\n%d, %d\n\n", t->size, t->uType);
    switch (t->uType) {
        case VECTOR: {
            FreeVector(t->vector);
            break;
        }
        case MATRIX2D: {
            FreeMatrix(t->matrix2d);
            break;
        }
        case MATRIX3D: {
            FreeMatrix3d(t->matrix3d);
            break;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in FreeTensor().", t->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
    free(t);
}

void AveragePoolMatrix(Matrix *mDst, Matrix *mSrc, unsigned int poolSize) {

    double* values = mDst->values;

    for (unsigned int r = 0; r<mSrc->r; r+= poolSize) {
        for (unsigned int c = 0; c<mSrc->r; c+= poolSize) {
            double max = GetMatrixValueRowCol(mSrc, r, c);
            for (int i = 0; i<poolSize*poolSize; i++) {
                if (GetMatrixValueRowCol(mSrc, r+(i/poolSize), c+(i%poolSize)) > max) {
                    max = GetMatrixValueRowCol(mSrc, r+(i/poolSize), c+(i%poolSize));
                }
            }
            *values = max;
            values++;
        }
    }
}

void MaxPoolMatrix(Matrix *mDst, Matrix *mSrc, unsigned int poolSize) {

    double* values = mDst->values;

    for (unsigned int r = 0; r<mSrc->r; r+= poolSize) {
        for (unsigned int c = 0; c<mSrc->r; c+= poolSize) {
            double sum = 0;
            for (int i = 0; i<poolSize*poolSize; i++) {
                sum += GetMatrixValueRowCol(mSrc, r+(i/poolSize), c+(i%poolSize));
            }
            *values = sum;
            values++;
        }
    }
}

void AveragePoolMatrix3d(Matrix3d *mDst, Matrix3d *mSrc, unsigned int poolSize) {

    double* values = mDst->values;

    for (int d = 0; d<mSrc->depth; d++) {
        for (unsigned int r = 0; r<mSrc->r; r+= poolSize) {
            for (unsigned int c = 0; c<mSrc->r; c+= poolSize) {
                double max = GetMatrix3DValueDepthRowCol(mSrc, d, r, c);
                for (int i = 0; i<poolSize*poolSize; i++) {
                    if (GetMatrix3DValueDepthRowCol(mSrc, d, r+(i/poolSize), c+(i%poolSize)) > max) {
                        max = GetMatrix3DValueDepthRowCol(mSrc, d, r+(i/poolSize), c+(i%poolSize));
                    }
                }
                *values = max;
                values++;
            }
        }
    }
}

void MaxPoolMatrix3d(Matrix3d *mDst, Matrix3d *mSrc, unsigned int poolSize) {

    double* values = mDst->values;

    for (int d = 0; d<mSrc->depth; d++) {
        for (unsigned int r = 0; r<mSrc->r; r+= poolSize) {
            for (unsigned int c = 0; c<mSrc->r; c+= poolSize) {
                double sum = 0;
                for (int i = 0; i<poolSize*poolSize; i++) {
                    sum += GetMatrix3DValueDepthRowCol(mSrc, d, r+(i/poolSize), c+(i%poolSize));
                }
                *values = sum;
                values++;
            }
        }
    }
}

/* debug function */
void PrintMatrix(Matrix *m) {
    char* buffer = malloc(MATRIX_PRINT_BUFFER * sizeof(char));

    printf("\n\n");
    for (int i = 0; i < m->r; i++) {
        strcpy(buffer, "| ");
        for (int j = 0; j < m->c; j++) {
            sprintf(buffer, "%s %+4.6f", buffer , GetMatrixValueRowCol(m, i, j));
        }
        sprintf(buffer, "%s  |\n", buffer);

        printf(buffer);
    }

    free(buffer);
}

/* debug function */
void PrintMatrix3dSlice(Matrix3d *m3d, unsigned int slice) {
    char* buffer = malloc(MATRIX_PRINT_BUFFER * sizeof(char));

    printf("\n\n");
    for (int i = 0; i < m3d->r; i++) {
        strcpy(buffer, "| ");
        for (int j = 0; j < m3d->c; j++) {
            sprintf(buffer, "%s %+4.6f", buffer , GetMatrix3DValueDepthRowCol(m3d, slice, i, j));
        }
        sprintf(buffer, "%s  |\n", buffer);

        printf(buffer);
    }

    free(buffer);
}

void ShadeMatrix(Matrix *m) {
    unsigned int barSize = m->c*3;
    char boxBar[barSize + 1];

    for (int a = 0; a<barSize; a++) {
        boxBar[a] = '-';
    }
    boxBar[barSize] = '\0';
    printf("\n\n/%s\\\n", boxBar);

    for (int i = 0; i<m->r; i++) {
        printf("|");
        for (int j = 0; j<m->c; j++) {
            char c = CharShader((unsigned char)GetMatrixValueRowCol(m, i, j));
            printf("%c%c%c", c, c, c);
        }
        printf("|\n");
    }

    printf("\\%s/\n", boxBar);
}

void PrintVectorVertical(Vector *v) {
    printf("\n\n");
    for (int i = 0; i < v->size; i++) {
        printf("| %+8.8f |\n", v->values[i]);
    }
}

void PrintVectorHorizontal(Vector *v) {
    printf("[%+8.5f", v->values[0]);
    for (int i = 1; i < v->size; i++) {
        printf(", %+8.5f", v->values[i]);
    }
    printf("]\n");
}

void WriteTensorInfo(Tensor *t, char *buffer) {
    switch (t->uType) {
        case VECTOR: {
            sprintf(buffer, "Vector - Size: %d", t->vector->size);
            break;
        }
        case MATRIX2D: {
            sprintf(buffer, "2D Matrix - Size: %dx%d", t->matrix2d->r, t->matrix2d->c);
            break;
        }
        case MATRIX3D: {
            sprintf(buffer, "3D Matrix - Size: %dx%dx%d", t->matrix3d->depth, t->matrix3d->r, t->matrix3d->c);
            break;
        }
    }
}