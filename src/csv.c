//
// Created by erenk on 7/28/2025.
//

#include "csv.h"

#include <stdlib.h>
#include <string.h>

#include "linearalgebra.h"

CSVFile* OpenCSVFile(char *path) {
    FILE* file = fopen(path, "r");

    unsigned int lines = CountLines(path);

    return NewCSVStruct(file, lines);
}

CSVFile* NewCSVStruct(FILE* file, unsigned int rows) {
    CSVFile *csv = malloc(sizeof(CSVFile));

    csv->file = file;
    csv->rows = rows;
    csv->lastLineRead = 0;

    return csv;
}
unsigned int GetNextLine(CSVFile* csvfile, char* buffer, int maxline) {

    if (csvfile->lastLineRead == csvfile->rows) {
        return -1;
    }

    fgets(buffer, maxline, csvfile->file);

    return ++csvfile->lastLineRead;
}

unsigned int CountLines(char *path) {
    FILE* file = fopen(path, "r");

    if (file == NULL) {
        return 0;
    }

    unsigned int lines = 0;
    int ch = 0;

    lines++;
    while ((ch = getc(file)) != EOF) {
        if (ch == '\n') {
            lines++;
        }
    }

    fclose(file);
    return lines;
}

void CSVInfo(CSVFile* csvfile) {
    printf("Rows: %d, Last line read: %d\n", csvfile->rows, csvfile->lastLineRead);
}

void FreeCSV(CSVFile* csvfile) {
    fclose(csvfile->file);
    free(csvfile);
}