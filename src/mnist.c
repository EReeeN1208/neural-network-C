//
// Created by erenk on 8/6/2025.
//

#include "mnist.h"

#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

#include "nn.h"
#include "util.h"


int ReadDigitFromCSV(CSVFile *csv, MnistDigit *d) {

    char lineBuffer[CSV_LINE_MAX_BUFF];
    char valueBuffer[CSV_VALUE_MAX_BUFF];

    int lineNum = GetNextLine(csv, lineBuffer);

    if (lineNum == -1) {
        return -1;
    }

    Vector *vLine = NewEmptyVector(1 + MNIST_PIXEL_COUNT);

    unsigned int lineLen = strlen(lineBuffer);
    unsigned int i = 0; // lineBuffer iterator;
    unsigned int j = 0; // valueBuffer iterator
    unsigned int k = 0; // vLine iterator

    while (i<lineLen && lineBuffer[i] != '\0') {
        valueBuffer[j++] = lineBuffer[i++];
        if (lineBuffer[i] == ',') {
            i++;
            valueBuffer[j] = '\0';
            j = 0;
            SetVectorValue(vLine, k++, StrToUChar(valueBuffer, CSV_VALUE_MAX_BUFF));
        }
    }

    d->digit = (char)GetVectorValue(vLine, 0);

    for (int px = 0; px<MNIST_PIXEL_COUNT; px++) {
        SetMatrixValuePos(d->pixels, px, GetVectorValue(vLine, px+1) / 255.0);
    }

    FreeVector(vLine);
    return 0;
}

MnistDigit* NewMnistDigit() {
    MnistDigit *d = malloc(sizeof(MnistDigit));
    d->pixels = NewEmptyMatrix(MNIST_DIGIT_SIDE_LEN, MNIST_DIGIT_SIDE_LEN);
    return d;
}

void TrainNetworkMNIST(NeuralNetwork *nNet, CSVFile *csvTrain, CSVFile *csvTest, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)) {
    if (nNet->stage < NET_LAYERS_FINALIZED) {
        fprintf(stderr, "Error: Attempted to train un-finalized neural network (stage %d). Required stage: >= 2", nNet->stage);
        exit(EXIT_FAILURE_CODE);
    }

    printf("Beginning to train network:\n");
    PrintNeuralNetworkInfo(nNet);

    if (nNet->highestTestResult == NULL) {
        nNet->highestTestResult = NewTestResultMNIST();
    }

    unsigned int trainingSteps = nNet->maxRoundSteps;

    if (trainingSteps == 0) {
        trainingSteps = csvTrain->rows-2;
    }
    if (trainingSteps>csvTrain->rows) {
        trainingSteps = csvTrain->rows-2;
    }

    MnistDigit *mnistDigit = NewMnistDigit();
    RewindCSV(csvTrain);

    for (int i = 0; i<nNet->trainingRounds; i++) {
        SkipLine(csvTrain);
        for (int j = 0; j<trainingSteps; j++) {
            ReadDigitFromCSV(csvTrain, mnistDigit);

            Tensor* inputTensor = NewTensorCloneMatrix(mnistDigit->pixels); //needs to be fred
            Tensor *netOutput = NetworkTrainingStep(nNet, inputTensor, mnistDigit->digit, lossFunction); //does not need to be freed. it's just a reference to the net's output tensor
            FreeTensor(inputTensor); //maybe optimize this later?

            if (nNet->trainingStepCount % MNIST_TEST_TRAINING_STEP_INTERVAL == 0) {
                RewindCSV(csvTest);
                SkipLine(csvTest);

                TestResult* testResult = TestNetworkMNIST(nNet, csvTest, CalculateLossMNIST);
                if (testResult->accuracy > nNet->highestTestResult->accuracy) {
                    FreeTestResult(nNet->highestTestResult);
                    nNet->highestTestResult = testResult;
                    nNet->highestTestResult->trainingStep = nNet->trainingStepCount;
                } else {
                    FreeTestResult(testResult);
                }
            }
        }
        RewindCSV(csvTrain);
    }

    FreeMnistDigit(mnistDigit);

    nNet->stage = NET_TRAINED;
}

TestResult* TestNetworkMNIST(NeuralNetwork  *nNet, CSVFile *csvTest, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)) {
    if (nNet->stage < NET_LAYERS_FINALIZED) {
        fprintf(stderr, "Error: Attempted to test un-finalized neural network (stage %d). Required stage: >= 1", nNet->stage);
        exit(EXIT_FAILURE_CODE);
    }
    /*
    if (nNet->stage < NET_TRAINED) {
        fprintf(stderr, "Warning: Attempted to test un-trained neural network (stage %d). Required stage: >= 3", nNet->stage);
        //exit(EXIT_FAILURE_CODE);
    }
    */

    MnistDigit *mnistDigit = NewMnistDigit();
    RewindCSV(csvTest);
    SkipLine(csvTest);
    unsigned int correct = 0;
    unsigned int incorrect = 0;

    TestResult* testResult = NewTestResultMNIST();

    for (int i = 0; i<csvTest->rows; i++) {
        ReadDigitFromCSV(csvTest, mnistDigit);

        Tensor* inputTensor = NewTensorCloneMatrix(mnistDigit->pixels); //needs to be fred
        Tensor *netOutput = RunNeuralNetwork(nNet, inputTensor); //don't free netOutput. just a reference

        double loss = lossFunction(netOutput, nNet->outputLayer->computedValueGradients, mnistDigit->digit);

        if (mnistDigit->digit == GetTensorMaxIndex(netOutput)) {
            correct++;
        } else {
            incorrect++;
        }

        IncrementMatrixValueRowCol(testResult->classificationMatrix, mnistDigit->digit, GetTensorMaxIndex(netOutput), 1);

        FreeTensor(inputTensor); //maybe optimize this later?
    }

    UpdateTestResult(testResult, correct, incorrect);

    FreeMnistDigit(mnistDigit);

    return testResult;
}

double CalculateLossMNIST(Tensor* probabilities, Tensor* outputGradient, int digit) {
    if (probabilities->size != MNIST_DIGIT_COUNT || outputGradient->size != MNIST_DIGIT_COUNT) {
        fprintf(stderr, "Error: tried to calculate MNIST loss for tensor with invalid sizes: %d, %d", probabilities->size, outputGradient->size);
        exit(EXIT_FAILURE_CODE);
    }
    // One possible function. Apparently, its not great
    /*
    return 1 - pow(GetTensorValuePos(probabilities, digit), 2);
    */


    // Mean Squared Error
    /*
    double totalError = 0;

    for (int i = 0; i<10; i++) {
        if (i != digit) {
            totalError += 0.5 * pow(0 - GetTensorValuePos(probabilities, i), 2);
        }
    }
    totalError += 0.5 * pow(1 - GetTensorValuePos(probabilities, digit), 2);

    return totalError;
    */


    // Cross Entropy (Apparently this is the best method for classification tasks like mnist)

    double oneHotVector[MNIST_DIGIT_COUNT] = { 0 };
    oneHotVector[digit] = 1;

    const double epsilon = 1e-15;
    double prob = GetTensorValuePos(probabilities, digit);
    prob = fmax(prob, epsilon);  // Ensure prob >= epsilon

    for (int i = 0; i < MNIST_DIGIT_COUNT; i++) {
        double gradient = GetTensorValues(probabilities)[i] - oneHotVector[i];
        // clamp gradients to prevent explosion
        //gradient = fmax(-10.0, fmin(10.0, gradient));
        GetTensorValues(outputGradient)[i] = gradient;
    }

    return -1.0 * log(prob); //return loss value. not really necessary
}

TestResult* NewTestResultMNIST() {
    TestResult* testResult = malloc(sizeof(TestResult));

    testResult->trainingStep = 0;
    testResult->totalCases = 0;
    testResult->correct = 0;
    testResult->incorrect = 0;
    testResult->accuracy = 0;

    testResult->classificationMatrix = NewEmptyMatrix(MNIST_DIGIT_COUNT, MNIST_DIGIT_COUNT);

    return testResult;
}

void FreeMnistDigit(MnistDigit *d) {
    if (d->pixels != NULL) {
        FreeMatrix(d->pixels);
    }
    free(d);
}

void PrintMnistDigit(MnistDigit *d) {
    ShadeMatrix(d->pixels);
}

CSVFile* GetMNISTTestCSV() {
    if (PLATFORM == PLATFORM_WINDOWS) {
        return OpenCSVFile("..\\data\\mnist_test.csv");
    } else {
        return OpenCSVFile("./../data/mnist_test.csv");
    }
}

CSVFile* GetMNISTTrainCSV() {
    if (PLATFORM == PLATFORM_WINDOWS) {
        return OpenCSVFile("..\\data\\mnist_train.csv");
    } else {
        return OpenCSVFile("./../data/mnist_train.csv");
    }
}
