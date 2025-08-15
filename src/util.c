//
// Created by Eren Kural on 11.08.2025.
//

#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

int StrToInt(char *s, unsigned int maxLen) {

    int i = 0;
    int out = 0;

    if (s[0] == '-') {
        i++;
    }

    while (i<maxLen && s[i] != '\0') {
        if ('0' > s[i] || s[i] > '9') {
            fprintf(stderr, "Error during str to int parsing. attempted to parse non-digit character %c\n", s[i]);
            exit(EXIT_FAILURE_CODE);
        }
        out*=10;
        out += (s[i++] - '0');
    }

    if (s[0] == '-') {
        out *= -1;
    }

    return out;
}

unsigned char StrToUChar(char *s, unsigned int maxLen) {

    int i = 0;
    unsigned char out = 0;

    while (i<maxLen && s[i] != '\0') {
        out*=10;
        out += (s[i++] - '0');
    }

    return out;
}

char CharShader(unsigned char c) {
    return SHADER_PALETTE[(c*SHADER_PALETTE_SIZE-1) / 255];
}

double GetRandomNormalised() {
    return pow((rand()%RAND_PRECISION / (double)RAND_PRECISION)*2-1, 3);
}