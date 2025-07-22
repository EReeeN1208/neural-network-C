#include <stdio.h>

#include "linearalgebra.h"

int main(void) {
    printf("Hello, World!\n");

    struct Matrix m1 = InitMatrix(5, 3, 3);
    struct Matrix m2 = InitMatrix(3, 4, 4);
    struct Matrix m3 = MatrixMultiply(&m1, &m2);

    PrintMatrix(&m3);


    struct Matrix m4 = InitMatrixIncremental(2, 4);
    struct Vector v1 = InitVector(4, 3);
    struct Vector v2 = VectorMatrixMultiply(&m4, &v1);

    PrintMatrix(&m4);
    PrintVector(&v1);
    PrintVector(&v2);

    struct Matrix m5 = GetIdentityMatrix(6);
    struct Matrix m6 = InitMatrixIncremental(6, 6);
    struct Matrix m7 = MatrixMultiply(&m5, &m6);

    PrintMatrix(&m5);
    PrintMatrix(&m6);
    PrintMatrix(&m7);

    struct Matrix m8 = TransposeMatrix(&m6);

    PrintMatrix(&m8);

    ScaleMatrixDouble(&m8, 2.5);

    PrintMatrix(&m8);

    return 0;
}