//
// Created by Eren Kural on 11.08.2025.
//

#ifndef UTIL_H
#define UTIL_H

#define SHADER_PALETTE " .-*=#"
#define SHADER_PALETTE_SIZE 6

int StrToInt(char *s, unsigned int maxLen);
unsigned char StrToUChar(char *s, unsigned int maxLen);

char CharShader(unsigned char c);

#endif //UTIL_H
