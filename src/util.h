//
// Created by Eren Kural on 11.08.2025.
//

#ifndef UTIL_H
#define UTIL_H

//https://paulbourke.net/dataformats/asciiart/#:~:text=%22-,.%3A%2D%3D%2B*%23%25%40,-%22
#define SHADER_PALETTE " .:-=+*#%@"
#define SHADER_PALETTE_SIZE 10

#define RAND_PRECISION 1000

#define EXIT_FAILURE_CODE 1


int StrToInt(char *s, unsigned int maxLen);
unsigned char StrToUChar(char *s, unsigned int maxLen);

char CharShader(unsigned char c);

double GetRandomNormalised();

#endif //UTIL_H
