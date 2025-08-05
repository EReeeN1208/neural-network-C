//
// Created by erenk on 7/28/2025.
//

#ifndef CSV_H
#define CSV_H

#include <stdio.h>

#include "linearalgebra.h"

typedef struct {
    FILE* file;
    int lastLineRead;
} CSVFile;

CSVFile* OpenFile(char *path);

#endif //CSV_H
