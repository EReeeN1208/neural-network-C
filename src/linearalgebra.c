//
// Created by Eren Kural on 19.07.2025.
//
#include "linearalgebra.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"


Vector* NewEmptyVector(unsigned int size) {
    Vector *v = malloc(sizeof(Vector));
    v->size = size;
    v->values = malloc(size * sizeof(double));

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
    free(v->values);
    free(v);
}



Matrix* NewEmptyMatrix(unsigned int r, unsigned int c) {
    Matrix *m = malloc(sizeof(Matrix));
    const unsigned int size = r * c;
    m->r = r;
    m->c = c;
    m->values = malloc(size * sizeof(double));

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
    Matrix *m = NewFilledMatrix(len, len, 0);
    for (unsigned i = 0; i < len*len; i += (len + 1)) {
        m->values[i] = 1;
    }
    return m;
}

void FreeMatrix(Matrix *m) {
    free(m->values);
    free(m);
}



Matrix3d* NewEmptyMatrix3d(unsigned int depth, unsigned int r, unsigned int c) {
    Matrix3d *m = malloc(sizeof(Matrix3d));
    const unsigned int size = depth * r * c;
    m->depth = depth;
    m->r = r;
    m->c = c;

    m->values = malloc(size * sizeof(double));

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
    for (int i = 0; i < vResult->size; i++) {
        sum = 0;
        for (int j = 0; j < m->c; j++) {
            sum += GetMatrixValueRowCol(m, i, j)*v->values[j];
        }
        vResult->values[i] = sum;
    }
    return vResult;
}

Matrix* MatrixAdd(Matrix *m1, Matrix *m2) {
    if (m1->r != m2->r || m1->c != m2->c) {
        fprintf(stderr, "Matrices not same size for addition");
    }
    int size = m1->r * m1->c;

    Matrix *mResult = NewEmptyMatrix(m1->r, m1->c);

    for (int i = 0; i < size; i++) {
        mResult->values[i] = m1->values[i] + m2->values[i];
    }
    return mResult;
}

Vector* VectorAdd(Vector *v1, Vector *v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Vectors not same size for addition");
    }

    Vector *vResult = NewEmptyVector(v1->size);

    for (int i = 0; i < v1->size; i++) {
        vResult->values[i] = v1->values[i] + v2->values[i];
    }
    return vResult;
}
Matrix* TransposeMatrix(Matrix *m) {
    unsigned int size = m->r * m->c;

    Matrix* mResult = NewEmptyMatrix(m->c, m->r);

    for (int i = 0; i < size; i++) {
        mResult->values[i] = GetMatrixValueRowCol(m, i % m->c, i / m->c);
    }
    return mResult;
}


void ScaleVectorDouble(Vector *v, double s) {
    for (int i = 0; i < v->size; i++) {
        v->values[i] = v->values[i] * s;
    }
}
void ScaleVectorInt(Vector *v, int s) {
    for (int i = 0; i < v->size; i++) {
        v->values[i] = v->values[i] * s;
    }
}
void ScaleMatrixDouble(Matrix *m, double s) {
    unsigned int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        m->values[i] = m->values[i] * s;
    }
}
void ScaleMatrixInt(Matrix *m, int s)  {
    unsigned int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        m->values[i] = m->values[i] * s;
    }
}
void ScaleMatrix3dDouble(Matrix3d *m3d, double s) {
    unsigned int size = m3d->depth * m3d->r * m3d->c;

    for (int i = 0; i < size; i++) {
        m3d->values[i] = m3d->values[i] * s;
    }
}
void ScaleMatrix3dInt(Matrix3d *m3d, int s) {
    unsigned int size = m3d->depth * m3d->r * m3d->c;

    for (int i = 0; i < size; i++) {
        m3d->values[i] = m3d->values[i] * s;
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



double GetMatrix3DValueRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c) {
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

void SetMatrix3DValueRowCol(Matrix3d *m3d, unsigned int depth, unsigned int r, unsigned int c, double value) {
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
    Vector *v = NewEmptyVector(m->r * m->c);
    v->values = m->values;
    free(m);
    return v;
}



Matrix* GetIdentityKernel(unsigned int size) {
    if (size % 2 != 1) {
        fprintf(stderr, "Invalid convolution identity matrix size %d", size);
        exit(EXIT_FAILURE_CODE);
    }
    Matrix *m = NewFilledMatrix(size, size, 0);
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



Tensor* NewTensorVector(Vector* v) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = VECTOR;
    t->vector = v;
    t->size = v->size;

    return t;
}
Tensor* NewTensorMatrix(Matrix* m) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX2D;
    t->matrix2d = m;
    t->size = m->r * m->r;

    return t;
}
Tensor* NewTensorMatrix3d(Matrix3d* m3d) {
    Tensor *t = malloc(sizeof(Tensor));
    t->uType = MATRIX3D;
    t->matrix3d = m3d;
    t->size = GetMatrix3dSize(m3d);

    return t;
}
Tensor* CloneTensorEmpty(Tensor *t) {
    Tensor *tNew = malloc(sizeof(Tensor));
    tNew->size = t->size;
    tNew->uType = t->uType;

    switch (t->uType) {
        case VECTOR: {
            tNew->vector = NewFilledVector(t->vector->size, 0);
            break;
        }
        case MATRIX2D: {
            tNew->matrix2d = NewFilledMatrix(t->matrix2d->r, t->matrix2d->c, 0);
            break;
        }
        case MATRIX3D: {
            tNew->matrix3d = NewFilledMatrix3d(t->matrix3d->depth, t->matrix3d->r, t->matrix3d->c, 0);
            break;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor CloneTensorEmpty().", t->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }

    return tNew; // does not leak even tough static analysis says it does
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
        };
    }
}

double GetTensorValuePos(Tensor *t, unsigned int pos) {
    if (pos >= t->size) {
        fprintf(stderr, "Error: tried to access out of bounds pos '%d' for a tensor. GetTensorValuePos().", pos);
        exit(EXIT_FAILURE_CODE);
    }
    return GetTensorValues(t)[pos];
}

void FreeTensor(Tensor *t) {
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


/* debug function */
void PrintMatrix(Matrix *m) {
    char* buffer = malloc(100 * sizeof(char));

    printf("\n\n");
    for (int i = 0; i < m->r; i++) {
        strcpy(buffer, "| ");
        for (int j = 0; j < m->c; j++) {
            sprintf(buffer, "%s %+8.4f", buffer , GetMatrixValueRowCol(m, i, j));
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
        printf("| %+8.4f |\n", v->values[i]);
    }
}

void PrintVectorHorizontal(Vector *v) {
    printf("\n[%+8.4f", v->values[1]);
    for (int i = 1; i < v->size; i++) {
        printf(", %+8.4f", v->values[i]);
    }
    printf("]\n");
}