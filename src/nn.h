//
// Created by erenk on 8/13/2025.
//

#ifndef NN_H
#define NN_H

#include "csv.h"
#include "linearalgebra.h"

// Trainable layers
#define LINEAR_LAYER 0
#define CONVOLUTION_LAYER 1

// IO Layers
#define VECTOR_LAYER 2
#define MATRIX_LAYER 3

// Non-Trainable layers
#define ELEMENTWISE_LAYER 4
#define POOLING_LAYER 5
// Pooling Layer Methods
#define MAX_POOLING 0
#define AVERAGE_POOLING 1

//Non-Parameterised layers
#define FLATTENING_LAYER 6
#define SOFTMAX_LAYER 7
#define RESHAPE_LAYER 8 //general name, but converts a 2d matrix into a 3d matrix


//Neural Network Stages
#define NET_EMPTY 0
#define NET_LAYERS_ADDED 1
#define NET_LAYERS_FINALIZED 2
#define NET_TRAINED 3

#define LAYER_WRITE_MAX_BUFF 30
#define NETWORK_PRINT_TRAINING_STEP_INTERVAL 100

//Note to self
//Weights format:
//First dimension (Rows) represents the node of the layer
//The second dimension (Columns) represents the value for each of the nodes in the previous layer

typedef struct {
    unsigned int size;
    Matrix *weights;
    Vector *biases;
    Matrix *weightGradients;
    Vector *biasGradients;
} LinearLayer;

typedef struct {
    unsigned int kernelCount;
    unsigned int kernelSize;
    Matrix **kernels;
    Matrix **kernelGradients;
    Vector *kernelBiases;
} ConvolutionLayer;


typedef struct {
    unsigned int size;
} VectorLayer; //for nNet io

typedef struct {
    unsigned int r;
    unsigned int c;
} MatrixLayer; //for nNet io


typedef struct {
    double (*function)(double);
    double (*gradientFunction)(double);
} ElementWiseLayer;

typedef struct {
    unsigned int poolSize;
    unsigned int poolingMethod;
} PoolingLayer;


typedef struct {

} FlatteningLayer;

typedef struct {

} SoftMaxLayer;

typedef struct {

} ReshapeLayer;


typedef struct {
    unsigned int uType;
    union {
        LinearLayer *linearLayer;
        ConvolutionLayer *convolutionLayer;

        VectorLayer *vectorLayer;
        MatrixLayer *matrixLayer;

        ElementWiseLayer *elementWiseLayer;
        PoolingLayer *poolingLayer;
        FlatteningLayer *flatteningLayer;
        SoftMaxLayer *softMaxLayer;
        ReshapeLayer *reshapeLayer;
    };
    Tensor *computedValues;
    Tensor *computedValueGradients;
} Layer;

typedef struct {
    unsigned int trainingStep;
    unsigned int totalCases;
    unsigned int correct;
    unsigned int incorrect;
    double accuracy; //0-1
    Matrix *classificationMatrix; // x/horizontal -> predicted class. y/vertical-> actual class
} TestResult;

typedef struct {

    char* name;

    Layer* inputLayer;
    Layer** hiddenLayers;
    Layer* outputLayer;

    unsigned int stage;
    unsigned int hiddenLayerCount;
    unsigned int trainingRounds;
    unsigned int maxRoundSteps; // 0 for unrestricted
    double learningRate;

    unsigned int trainingStepCount;

    TestResult* highestTestResult;

} NeuralNetwork;

Layer* NewLinearLayer(unsigned int size);
void FreeLinearLayer(Layer *layer);

Layer* NewConvolutionLayer(unsigned int kernelCount, unsigned int kernelSize);
void FreeConvolutionLayer(Layer *layer);


Layer* NewVectorLayer(unsigned int size);
void FreeVectorLayer(Layer *layer);

Layer* NewMatrixLayer(unsigned int r, unsigned int c);
void FreeMatrixLayer(Layer *layer);


Layer* NewElementWiseLayer(double (*function)(double), double (*gradientFunction)(double));
void FreeElementWiseLayer(Layer *layer);

Layer* NewPoolingLayer(unsigned int poolSize, unsigned int poolingMethod);
void FreePoolingLayer(Layer *layer);

Layer* NewFlatteningLayer(void);
void FreeFlatteningLayer(Layer *layer);

Layer* NewSoftMaxLayer(void);
void FreeSoftMaxLayer(Layer *layer);

Layer* NewReshapeLayer(void);
void FreeReshapeLayer(Layer *layer);

void FreeLayer(Layer *layer);
void PrintLayerInfo(Layer *layer);

void InitIOLayerTensor(Layer *layer);
void InitHiddenLayerTensor(Layer *previousLayer, Layer *layer);
void ComputeLayerValues(Layer *previousLayer, Layer *layer);

void ComputeHiddenLayerGradients(Layer *previousLayer, Layer *layer, Layer *proceedingLayer, double lr);
void ComputeIOLayerGradients(Layer *previousLayer, Layer *layer);

NeuralNetwork* NewNeuralNetwork(unsigned int trainingRounds, unsigned int maxRoundSteps, double learningRate, char *name);
void PrintNeuralNetworkInfo(NeuralNetwork *nNet);
void FreeNeuralNetwork(NeuralNetwork *nNet);

void AddHiddenLayer(NeuralNetwork *nNet, Layer *layer);
void SetInputLayer(NeuralNetwork *nNet, Layer *layer);
void SetOutputLayer(NeuralNetwork *nNet, Layer *layer);

void FinalizeNeuralNetworkLayers(NeuralNetwork *nNet); //Call after adding all layers

Tensor* RunNeuralNetwork(NeuralNetwork *nNet, Tensor *input); //Do not free the output tensor
Tensor* NetworkTrainingStep(NeuralNetwork *nNet, Tensor *input, int expected, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)); //Do not free the output tensor

void UpdateTestResult(TestResult* testResult, unsigned int correct, unsigned int incorrect);
void PrintTestResult(TestResult* testResult);
void FreeTestResult(TestResult* testResult);


#endif //NN_H
