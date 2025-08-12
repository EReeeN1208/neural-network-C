#include "tests.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csv.h"
#include "linearalgebra.h"
#include "mnist.h"
#include "util.h"

//const int MAX_BUFF = 5000;

int RunAllTests(void) {
    return TestUtil() + TestMnist() + TestCSV() + TestLinearAlgebra();
}

int TestUtil(void) {
    unsigned int maxLen = 20;

    char buff[maxLen];

    printf(">%c<   >%c<   >%c<   >%c<   >%c<\n", CharShader(0), CharShader(50), CharShader(100), CharShader(240), CharShader(255));

    for (int i = 0; i < 256; i++) {
        printf("%c", CharShader(i));
    }
    printf("\n");

    strcpy(buff, "127");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));

    strcpy(buff, "-2372");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));

    strcpy(buff, "01270");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));


    strcpy(buff, "127");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "-2372");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "0261");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "0255");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    return 0;
}

int TestMnist(void) {

    CSVFile *csv;

    // Windows
    // csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];

    SkipLine(csv);

    MnistDigit *d = NewMnistDigit();

    ReadDigitFromCSV(csv, d);

    printf("%d\n", d->digit);

    PrintMatrix(d->pixels);

    PrintMnistDigit(d);

    return 0;
}

int TestCSV(void) {

    CSVFile *csv;

    // Windows
    // csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];


    while (GetNextLine(csv, buffer) != -1) {
        ++count;
        printf("Loop ran %d times\n", count);
        //printf("line %d: %s\n", count, buffer);
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
    Vector *v3 = NewIncrementalVector(37);
    Vector *v4 = GetSubVector(v3, 1, 37);
    Vector *v5 = GetSubVector(v3, 5, 10);

    PrintMatrix(m4);
    PrintVector(v1);
    PrintVector(v2);
    PrintVector(v3);
    PrintVector(v4);
    PrintVector(v5);

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