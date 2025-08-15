#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linearalgebra.h"
#include "activationFunctions.h"
#include "nn.h"
#include "tests.h"

int main(void) {

    srand(time(NULL));

    //printf("%lu, %lu, %lu", sizeof(double), sizeof(int), sizeof(long));

    /*
    for (int i = 0; i<25; i++) {

        printf("| %d, %d ", i/5 - 2, i%5 - 2);
        if (i % 5 == 4) {
            printf("|\n");
        }
    }
    */

    //printf("%f, %f, %f\n", (rand()%1000 / (double)1000)*2-1, (rand()%1000 / (double)1000)*2-1, (rand()%1000 / (double)1000)*2-1);

    //TestLinearAlgebra();

    //TestMnist();
    //TestConvolution();
    NeuralNetworkMain();
}