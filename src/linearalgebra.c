//
// Created by Eren Kural on 19.07.2025.
//
#include "linearalgebra.h"

#include <stdlib.h>

struct Matrix NewMatrix(unsigned int h, unsigned int w) {
    struct Matrix m;
    const unsigned int size = h * w;
    m.h = h;
    m.w = w;
    m.values = malloc(size * sizeof(unsigned int));

    return m;
}

struct Matrix InitMatrix(unsigned int h, unsigned int w, double fill) {
    struct Matrix m = NewMatrix(h, w);
    const unsigned int size = h * w;
    double* vp = m.values;

    for (int i = 0; i < size; i++) {
        *(vp++) = fill;
    }

    return m;
}

void DelMatrix(struct Matrix *m) {
    free(m->values);
}

struct Vector NewVector(unsigned int size) {
    struct Vector v;
    v.size = size;
    v.values = malloc(size * sizeof(unsigned int));

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

void DelVector(struct Vector *v) {
    free(v->values);
}

struct Matrix MatrixMultiply(struct Matrix m1, struct Matrix m2);
struct Matrix VectorMatrixMultiply(struct Matrix m, struct Vector v);

struct Matrix Transpose(struct Matrix m);

struct Matrix ScaleMatrixDouble(struct Matrix m, double s);
struct Vector ScaleVectorDouble(struct Vector m, double s);
struct Matrix ScaleMatrixInt(struct Matrix m, int s);
struct Vector ScaleVectorInt(struct Vector m, int s);