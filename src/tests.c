#include "tests.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activationFunctions.h"
#include "csv.h"
#include "linearalgebra.h"
#include "mnist.h"
#include "util.h"

int TestLinearNeuralNetworkMNIST(void) {

    NeuralNetwork *nNet = NewNeuralNetwork(MNIST_TRAINING_ROUNDS, MNIST_MAX_TRAINING_STEPS, MNIST_LEARNING_RATE, "MNIST Linear Test");

    SetInputLayer(nNet, NewMatrixLayer(28, 28));

    AddHiddenLayer(nNet, NewReshapeLayer());
    AddHiddenLayer(nNet, NewPoolingLayer(2, AVERAGE_POOLING));
    AddHiddenLayer(nNet, NewFlatteningLayer());
    AddHiddenLayer(nNet, NewLinearLayer(512));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(256));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(128));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(10));
    AddHiddenLayer(nNet, NewSoftMaxLayer());

    SetOutputLayer(nNet, NewVectorLayer(10));

    FinalizeNeuralNetworkLayers(nNet);

    // CSV-MNIST
    CSVFile *csvTrain = GetMNISTTrainCSV(); //csv file with data used to train the network (size = 60000)
    CSVFile *csvTest = GetMNISTTestCSV(); //csv file with data used to test the trained network (size = 10000)

    TrainNetworkMNIST(nNet, csvTrain, csvTest, CalculateLossMNIST);
    PrintNeuralNetworkInfo(nNet);
    TestResult* testResult = TestNetworkMNIST(nNet, csvTest, CalculateLossMNIST);

    printf("Final Test Result:\n");
    PrintTestResult(testResult);
    printf("\nHighest Test Result:\n");
    PrintTestResult(nNet->highestTestResult);

    //Cleanup
    FreeTestResult(testResult);
    FreeNeuralNetwork(nNet);
    FreeCSV(csvTrain);
    FreeCSV(csvTest);

    return 0;
}
int TestConvolutionalNeuralNetworkMNIST(void) {

    NeuralNetwork *nNet = NewNeuralNetwork(MNIST_TRAINING_ROUNDS, MNIST_MAX_TRAINING_STEPS, MNIST_LEARNING_RATE, "MNIST Convolutional Test");

    SetInputLayer(nNet, NewMatrixLayer(28, 28));

    AddHiddenLayer(nNet, NewReshapeLayer());
    AddHiddenLayer(nNet, NewConvolutionLayer(16, 3));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewPoolingLayer(2, MAX_POOLING));
    AddHiddenLayer(nNet, NewFlatteningLayer());
    AddHiddenLayer(nNet, NewLinearLayer(256));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(10));
    AddHiddenLayer(nNet, NewSoftMaxLayer());

    SetOutputLayer(nNet, NewVectorLayer(10));

    FinalizeNeuralNetworkLayers(nNet);

    // CSV-MNIST
    CSVFile *csvTrain = GetMNISTTrainCSV(); //csv file with data used to train the network (size = 60000)
    CSVFile *csvTest = GetMNISTTestCSV(); //csv file with data used to test the trained network (size = 10000)

    TrainNetworkMNIST(nNet, csvTrain, csvTest, CalculateLossMNIST);
    PrintNeuralNetworkInfo(nNet);
    TestResult* testResult = TestNetworkMNIST(nNet, csvTest, CalculateLossMNIST);

    printf("Final Test Result:\n");
    PrintTestResult(testResult);
    printf("\nHighest Test Result:\n");
    PrintTestResult(nNet->highestTestResult);

    //Cleanup
    FreeTestResult(testResult);
    FreeNeuralNetwork(nNet);
    FreeCSV(csvTrain);
    FreeCSV(csvTest);

    return 0;
}

int RunAllTests(void) {
    return TestConvolution() + TestUtil() + TestMnist() + TestCSV() + TestLinearAlgebra();
}

int TestConvolution(void) {
    CSVFile *csv;

    // Windows
    // csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];

    SkipLine(csv);

    MnistDigit *d = NewMnistDigit();
    ReadDigitFromCSV(csv, d);

    Matrix *identityKernel = GetIdentityKernel(3);
    Matrix *edgeDetectionKernel = GetEdgeDetectionKernel();
    Matrix *blurKernel = GetBlurKernel(5);

    PrintMatrix(identityKernel);
    PrintMatrix(edgeDetectionKernel);
    PrintMatrix(blurKernel);

    Matrix *dIdentity = ConvolveMatrix(d->pixels, identityKernel);
    Matrix *dEdgeDetection = ConvolveMatrix(d->pixels, edgeDetectionKernel);
    Matrix *dBlur = ConvolveMatrix(d->pixels, blurKernel);

    PrintMnistDigit(d);
    ShadeMatrix(d->pixels);
    ShadeMatrix(dIdentity);
    ShadeMatrix(dEdgeDetection);
    ShadeMatrix(dBlur);

    FreeMnistDigit(d);
    FreeCSV(csv);

    FreeMatrix(identityKernel);
    FreeMatrix(edgeDetectionKernel);
    FreeMatrix(blurKernel);

    FreeMatrix(dIdentity);
    FreeMatrix(dEdgeDetection);
    FreeMatrix(dBlur);


    return 0;
}

int TestUtil(void) {
    unsigned int maxLen = 20;

    char buff[maxLen];

    printf(">%c<   >%c<   >%c<   >%c<   >%c<\n", CharShader(0), CharShader(50), CharShader(100), CharShader(240), CharShader(255));

    for (int i = 0; i < 256; i++) {
        printf("%c", CharShader(i));
    }
    printf("\n");

    strcpy(buff, "127");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));

    strcpy(buff, "-2372");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));

    strcpy(buff, "01270");
    printf("%s, %d\n", buff, StrToInt(buff, maxLen));


    strcpy(buff, "127");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "-2372");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "0261");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    strcpy(buff, "0255");
    printf("%s, %d\n", buff, StrToUChar(buff, maxLen));

    return 0;
}

int TestMnist(void) {

    CSVFile *csv;

    // Windows
    // csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];

    SkipLine(csv);

    MnistDigit *d = NewMnistDigit();
    /*
    ReadDigitFromCSV(csv, d);

    printf("%d\n", d->digit);

    PrintMatrix(d->pixels);

    PrintMnistDigit(d);
    */
    for (int i = 0; i<5; i++) {
        ReadDigitFromCSV(csv, d);
        PrintMnistDigit(d);
    }

    FreeMnistDigit(d);
    FreeCSV(csv);

    return 0;
}

int TestCSV(void) {

    CSVFile *csv;

    // Windows
    // csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];


    while (GetNextLine(csv, buffer) != -1) {
        ++count;
        printf("Loop ran %d times\n", count);
        //printf("line %d: %s\n", count, buffer);
    }

    FreeCSV(csv);

    return 0;
}

int TestLinearAlgebra(void) {
    printf("Hello, World!\n");

    Matrix *m1 = NewFilledMatrix(5, 3, 3);
    Matrix *m2 = NewFilledMatrix(3, 4, 4);
    Matrix *m3 = MatrixMultiply(m1, m2);

    PrintMatrix(m3);


    Matrix *m4 = NewIncrementalMatrix(2, 4);
    Vector *v1 = NewFilledVector(4, -3);
    Vector *v2 = VectorMatrixMultiply(m4, v1);
    Vector *v3 = NewIncrementalVector(37);
    Vector *v4 = GetSubVector(v3, 1, 36);
    Vector *v5 = GetSubVector(v3, 5, 10);

    PrintMatrix(m4);
    PrintVectorVertical(v1);
    PrintVectorVertical(v2);
    PrintVectorVertical(v3);
    PrintVectorVertical(v4);
    PrintVectorVertical(v5);

    Matrix *m5 = GetIdentityMatrix(6);
    Matrix *m6 = NewIncrementalMatrix(6, 6);
    Matrix *m7 = MatrixMultiply(m5, m6);

    PrintMatrix(m5);
    PrintMatrix(m6);
    PrintMatrix(m7);

    Matrix *m8 = TransposeMatrix(m6);

    PrintMatrix(m8);

    ScaleMatrixDouble(m8, 2.5);

    PrintMatrix(m8);

    Matrix *m9 = NewRandomisedMatrix(10, 10);
    PrintMatrix(m9);


    return 0;
}
