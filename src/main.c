#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linearalgebra.h"
#include "activationfunctions.h"
#include "nn.h"
#include "tests.h"

int main(void) {

    srand(time(NULL));

    printf("Choose an option:\n");
    printf("0 - Linear MNIST Neural Network\n");
    printf("1 - Convolutional MNIST Neural Network\n");
    printf(">>> ");

    int c = getchar();
    printf("\n");

    switch (c) {
        case '0': {
            TestLinearNeuralNetworkMNIST();
            break;
        }
        case '1': {
            TestConvolutionalNeuralNetworkMNIST();
            break;
        }
        default: {
            printf("invalid choice");
            break;
        }
    }

}