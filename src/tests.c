#include <stdio.h>
#include <stdlib.h>

#include "csv.h"
#include "linearalgebra.h"

const int MAX_BUFF = 5000;

int TestCSV(void) {


    CSVFile *csv = OpenCSVFile("..\\data\\mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[MAX_BUFF];

    while (GetNextLine(csv, buffer, MAX_BUFF) != -1) {
        printf("Loop ran %d times\n", ++count);
    }

    FreeCSV(csv);

    return 0;
}

int TestLinearAlgebra(void) {
    printf("Hello, World!\n");

    Matrix *m1 = NewFilledMatrix(5, 3, 3);
    Matrix *m2 = NewFilledMatrix(3, 4, 4);
    Matrix *m3 = MatrixMultiply(m1, m2);

    PrintMatrix(m3);


    Matrix *m4 = NewIncrementalMatrix(2, 4);
    Vector *v1 = NewFilledVector(4, 3);
    Vector *v2 = VectorMatrixMultiply(m4, v1);

    PrintMatrix(m4);
    PrintVector(v1);
    PrintVector(v2);

    Matrix *m5 = GetIdentityMatrix(6);
    Matrix *m6 = NewIncrementalMatrix(6, 6);
    Matrix *m7 = MatrixMultiply(m5, m6);

    PrintMatrix(m5);
    PrintMatrix(m6);
    PrintMatrix(m7);

    Matrix *m8 = TransposeMatrix(m6);

    PrintMatrix(m8);

    ScaleMatrixDouble(m8, 2.5);

    PrintMatrix(m8);

    return 0;
}