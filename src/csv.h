//
// Created by erenk on 7/28/2025.
//

#ifndef CSV_H
#define CSV_H

#include <stdio.h>

#include "linearalgebra.h"



typedef struct {
    FILE* file;
    unsigned int rows;
    unsigned int lastLineRead;
} CSVFile;

CSVFile* OpenCSVFile(char *path);
CSVFile* NewCSVStruct(FILE* file, unsigned int rows);
unsigned int GetNextLine(CSVFile* csvfile, char* buffer, int maxline);
unsigned int CountLines(char *path);

void CSVInfo(CSVFile* csvfile);
void FreeCSV(CSVFile* csvfile);

#endif //CSV_H
