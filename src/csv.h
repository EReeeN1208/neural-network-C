//
// Created by erenk on 7/28/2025.
//

#ifndef CSV_H
#define CSV_H

#include <stdio.h>

#include "linearalgebra.h"

#define CSV_LINE_MAX_BUFF 5000
#define CSV_VALUE_MAX_BUFF 10



typedef struct {
    FILE* file;
    unsigned int rows;
    unsigned int lastLineRead;
} CSVFile;

CSVFile* OpenCSVFile(char *path);
CSVFile* NewCSV(FILE* file, unsigned int rows);
int GetNextLine(CSVFile* csvfile, char* buffer);
int SkipLine(CSVFile* csvfile); //Use to skip header
unsigned int CountLines(char *path);

void CSVInfo(CSVFile* csvfile);
void FreeCSV(CSVFile* csvfile);

#endif //CSV_H
