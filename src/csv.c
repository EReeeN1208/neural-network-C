//
// Created by erenk on 7/28/2025.
//

#include "csv.h"

#include <stdlib.h>


CSVFile* OpenCSVFile(char *path) {
    FILE* file = fopen(path, "r");

    unsigned int lines = CountLines(path);

    return NewCSV(file, lines);
}

CSVFile* NewCSV(FILE* file, unsigned int rows) {
    CSVFile *csv = malloc(sizeof(CSVFile));

    csv->file = file;
    csv->rows = rows;
    csv->lastLineRead = 0;

    return csv;
}

// Returns no# line read (counting from 1)
int GetNextLine(CSVFile* csvfile, char* buffer) {

    if (csvfile->lastLineRead == csvfile->rows) {
        return -1;
    }

    fgets(buffer, CSV_LINE_MAX_BUFF, csvfile->file);

    return (int)++csvfile->lastLineRead;
}

int SkipLine(CSVFile* csvfile) {

    if (csvfile->lastLineRead == csvfile->rows) {
        return -1;
    }

    char buff[CSV_LINE_MAX_BUFF];

    fgets(buff, CSV_LINE_MAX_BUFF, csvfile->file);

    return (int)++csvfile->lastLineRead;
}

void RewindCSV(CSVFile* csvfile) {
    rewind(csvfile->file);
    csvfile->lastLineRead = 0;
}

unsigned int CountLines(char *path) {
    FILE* file = fopen(path, "r");

    if (file == NULL) {
        fprintf(stderr, "unable to read csv file located at %s", path);
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