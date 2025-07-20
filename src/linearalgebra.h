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
    unsigned int h, w;
};

struct Matrix NewMatrix(unsigned int h, unsigned int w);
struct Matrix InitMatrix(unsigned int h, unsigned int w, double fill);
void DelMatrix(struct Matrix *m);

struct Vector NewVector(unsigned int size);
struct Vector InitVector(unsigned int size, double fill);
void DelVector(struct Vector *v);

struct Matrix MatrixMultiply(struct Matrix m1, struct Matrix m2);
struct Matrix VectorMatrixMultiply(struct Matrix m, struct Vector v);

struct Matrix Transpose(struct Matrix m);

struct Matrix ScaleMatrixDouble(struct Matrix m, double s);
struct Vector ScaleVectorDouble(struct Vector m, double s);
struct Matrix ScaleMatrixInt(struct Matrix m, int s);
struct Vector ScaleVectorInt(struct Vector m, int s);

#endif //LINEARALGEBRA_H


