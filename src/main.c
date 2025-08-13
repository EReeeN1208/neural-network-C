#include <stdio.h>

#include "linearalgebra.h"
#include "activationFunctions.h"
#include "tests.h"

int main(void) {

    //printf("%lu, %lu, %lu", sizeof(double), sizeof(int), sizeof(long));

    /*
    for (int i = 0; i<25; i++) {

        printf("| %d, %d ", i/5 - 2, i%5 - 2);
        if (i % 5 == 4) {
            printf("|\n");
        }
    }
    */

    //TestMnist();
    TestConvolution();
}