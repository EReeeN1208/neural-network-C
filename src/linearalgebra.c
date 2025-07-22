//
// Created by Eren Kural on 19.07.2025.
//
#include "linearalgebra.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Matrix NewMatrix(unsigned int r, unsigned int c) {
    struct Matrix m;
    const unsigned int size = r * c;
    m.r = r;
    m.c = c;
    m.values = malloc(size * sizeof(double));

    return m;
}

struct Matrix InitMatrix(unsigned int r, unsigned int c, double fill) {
    struct Matrix m = NewMatrix(r, c);

    const unsigned int size = r * c;
    double* vp = m.values;

    for (int i = 0; i < size; i++) {
        *(vp++) = fill;
    }


    return m;
}

struct Matrix InitMatrixIncremental(unsigned int r, unsigned int c) {
    struct Matrix m = NewMatrix(r, c);

    const unsigned int size = r * c;

    for (int i = 0; i < size; i++) {
        m.values[i] = i;
    }
    return m;
}

struct Matrix GetIdentityMatrix(unsigned int len) {
    struct Matrix m = InitMatrix(len, len, 0);
    for (unsigned i = 0; i < len*len; i += (len + 1)) {
        m.values[i] = 1;
    }
    return m;
}

void DelMatrix(struct Matrix *m) {
    free(m->values);
}

struct Vector NewVector(unsigned int size) {
    struct Vector v;
    v.size = size;
    v.values = malloc(size * sizeof(double));

    return v;
}
struct Vector InitVector(unsigned int size, double fill) {
    struct Vector v = NewVector(size);
    double* vp = v.values;

    for (int i = 0; i < size; i++) {
        *(vp++) = fill;
    }

    return v;
}

struct Vector InitVectorIncremental(unsigned int size) {
    struct Vector v = NewVector(size);

    for (int i = 0; i < size; i++) {
        v.values[i] = i;
    }
    return v;
}

void DelVector(struct Vector *v) {
    free(v->values);
}

struct Matrix MatrixMultiply(struct Matrix *m1, struct Matrix *m2) {
    if (m1->c != m2->r) {
        fprintf(stderr, "Error during matrix matrix multiplication. Sizes: %dx%d and %dx%d", m1->r, m1->c, m2->r, m2->c);
        exit(-1);
    }
    int size = m1->r * m2->c;
    struct Matrix mResult = NewMatrix(m1->r, m2->c);

    double sum;
    int r, c;
    for (int i = 0; i < size; i++) {
        r = i / mResult.c;
        c = i % mResult.c;
        sum = 0;

        for (int j = 0; j < m1->c; j++) {
            sum += GetMatrixValue(m1, r, j) * GetMatrixValue(m2, j, c);
        }
        mResult.values[i] = sum;
    }

    return mResult;
}

struct Vector VectorMatrixMultiply(struct Matrix *m, struct Vector *v) {
    if (m->c != v->size) {
        fprintf(stderr, "Error during matrix vector multiplication. Sizes: %dx%d and v%d", m->r, m->c, v->size);
        exit(-1);
    }

    struct Vector vResult = NewVector(m->r);

    double sum;
    for (int i = 0; i < vResult.size; i++) {
        sum = 0;
        for (int j = 0; j < m->c; j++) {
            sum += GetMatrixValue(m, i, j)*v->values[j];
        }
        vResult.values[i] = sum;
    }
    return vResult;
}

struct Matrix MatrixAdd(struct Matrix *m1, struct Matrix *m2) {
    if (m1->r != m2->r || m1->c != m2->c) {
        fprintf(stderr, "Matrices not same size for addition");
    }
    int size = m1->r * m1->c;

    struct Matrix m = NewMatrix(m1->r, m1->c);

    for (int i = 0; i < size; i++) {
        m.values[i] = m1->values[i] + m2->values[i];
    }
    return m;
}
struct Vector VectorAdd(struct Vector *v1, struct Vector *v2) {
    if (v1->size != v2->size) {
        fprintf(stderr, "Vectors not same size for addition");
    }

    struct Vector v = NewVector(v1->size);

    for (int i = 0; i < v1->size; i++) {
        v.values[i] = v1->values[i] + v2->values[i];
    }
    return v;
}

struct Matrix TransposeMatrix(struct Matrix *m) {
    int size = m->r * m->c;

    struct Matrix mResult = NewMatrix(m->c, m->r);

    for (int i = 0; i < size; i++) {
        mResult.values[i] = GetMatrixValue(m, i % m->c, i / m->c);
    }
    return mResult;
}

void ScaleMatrixDouble(struct Matrix *m, double s) {
    int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        m->values[i] = m->values[i] * s;
    }
}
void ScaleVectorDouble(struct Vector *v, double s) {
    for (int i = 0; i < v->size; i++) {
        v->values[i] = v->values[i] * s;
    }
}
void ScaleMatrixInt(struct Matrix *m, int s)  {
    int size = m->r * m->c;

    for (int i = 0; i < size; i++) {
        m->values[i] = m->values[i] * s;
    }
}
void ScaleVectorInt(struct Vector *v, int s) {
    for (int i = 0; i < v->size; i++) {
        v->values[i] = v->values[i] * s;
    }
}

/* Start counting from 0*/
double GetMatrixValue(struct Matrix *m, unsigned int r, unsigned int c) {
    if (r >= m->r || c >= m->c) {
        fprintf(stderr, "Error during matrix value access. Matrix size: %dx%d Tried to access: %dx%d", m->r, m->c, r, c);
        exit(-1);
    }
    return m->values[r*m->c + c];
}

/* Start counting from 0*/
void SetMatrixValue(struct Matrix *m, unsigned int r, unsigned int c, double value) {
    if (r >= m->r || c >= m->c) {
        fprintf(stderr, "Error during matrix value set. Matrix size: %dx%d Tried to set: %dx%d", m->r, m->c, r, c);
        exit(-1);
    }
    m->values[r*m->c + c] = value;
}

/* debug function */
void PrintMatrix(struct Matrix *m) {
    char* buffer = malloc(100 * sizeof(char));

    printf("\n\n");
    for (int i = 0; i < m->r; i++) {
        strcpy(buffer, "| ");
        for (int j = 0; j < m->c; j++) {
            sprintf(buffer, "%s %8.4f", buffer , GetMatrixValue(m, i, j));
        }
        sprintf(buffer, "%s  |\n", buffer);

        printf(buffer);
    }

    free(buffer);
}

void PrintVector(struct Vector *v) {
    printf("\n\n");
    for (int i = 0; i < v->size; i++) {
        printf("| %8.4f |\n", v->values[i]);
    }
}