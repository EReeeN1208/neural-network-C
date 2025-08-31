#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linearalgebra.h"
#include "activationfunctions.h"
#include "nn.h"
#include "tests.h"

int main(void) {

    srand(time(NULL));

    TestConvolutionalNeuralNetworkMNIST();
}