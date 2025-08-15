//
// Created by erenk on 8/6/2025.
//

#include "mnist.h"

#include <stdlib.h>
#include <string.h>

#include "util.h"


int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d) {

    char lineBuffer[CSV_LINE_MAX_BUFF];
    char valueBuffer[CSV_VALUE_MAX_BUFF];

    int lineNum = GetNextLine(csv, lineBuffer);

    if (lineNum == -1) {
        return -1;
    }

    Vector *vLine = NewEmptyVector(1 + MNIST_PIXEL_COUNT);

    unsigned int lineLen = strlen(lineBuffer);
    unsigned int i = 0; // lineBuffer iterator;
    unsigned int j = 0; // valueBuffer iterator
    unsigned int k = 0; // vLine iterator

    while (i<lineLen && lineBuffer[i] != '\0') {
        valueBuffer[j++] = lineBuffer[i++];
        if (lineBuffer[i] == ',') {
            i++;
            valueBuffer[j] = '\0';
            j = 0;
            SetVectorValue(vLine, k++, StrToUChar(valueBuffer, CSV_VALUE_MAX_BUFF));
        }
    }

    d->digit = (char)GetVectorValue(vLine, 0);

    for (int px = 0; px<MNIST_PIXEL_COUNT; px++) {
        SetMatrixValuePos(d->pixels, px, GetVectorValue(vLine, px+1));
    }

    FreeVector(vLine);
    return 0;
}

MnistDigit* NewMnistDigit() {
    MnistDigit *d = malloc(sizeof(MnistDigit));
    d->pixels = NewEmptyMatrix(MNIST_DIGIT_SIDE_LEN, MNIST_DIGIT_SIDE_LEN);
    return d;
}

void FreeMnistDigit(MnistDigit *d) {
    FreeMatrix(d->pixels);
    free(d);
}

void PrintMnistDigit(MnistDigit *d) {
    /*
    printf("\n\n/- digit: %d -------------------------------------------------------------------------\\\n", d->digit);

    for (int i = 0; i<MNIST_DIGIT_SIDE_LEN; i++) {
        printf("|");
        for (int j = 0; j<MNIST_DIGIT_SIDE_LEN; j++) {
            char c = CharShader((unsigned char)GetMatrixValueRowCol(d->pixels, i, j));
            printf("%c%c%c", c, c, c);
        }
        printf("|\n");
    }

    printf("\\------------------------------------------------------------------------------------/\n");
    */
    ShadeMatrix(d->pixels);
}
