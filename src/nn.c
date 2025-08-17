//
// Created by erenk on 8/13/2025.
//

#include "nn.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activationFunctions.h"
#include "csv.h"
#include "mnist.h"
#include "util.h"


/*
 * Trainable layers
 */
Layer* NewLinearLayer(unsigned int size) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = LINEAR_LAYER;
    layer->linearLayer = malloc(sizeof(LinearLayer));
    layer->linearLayer->size = size;
    layer->linearLayer->weights = NULL; // will be allocated during InitNeuralNetworkLayers()
    layer->linearLayer->biases = NewFilledVector(size, 0);

    layer->computedValues = NULL;

    return layer;
}
void FreeLinearLayer(Layer *layer) {
    FreeVector(layer->linearLayer->biases);
    FreeMatrix(layer->linearLayer->weights);
    free(layer->linearLayer);
    free(layer);
}

Layer* NewConvolutionLayer(unsigned int kernelCount, unsigned int kernelSize) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = CONVOLUTION_LAYER;
    layer->convolutionLayer = malloc(sizeof(ConvolutionLayer));
    layer->convolutionLayer->kernelCount = kernelCount;
    layer->convolutionLayer->kernelSize = kernelSize;
    layer->convolutionLayer->kernels = malloc(sizeof(Matrix*) * kernelCount);
    for (int i = 0; i<kernelCount; i++) {
        layer->convolutionLayer->kernels[i] = NewRandomisedMatrix(kernelSize, kernelSize);
    }

    layer->computedValues = NULL;

    return layer;
}
void FreeConvolutionLayer(Layer *layer) {
    for (int i = 0; i<layer->convolutionLayer->kernelCount; i++) {
        FreeMatrix(layer->convolutionLayer->kernels[i]);
    }
    free(layer->convolutionLayer);
    free(layer);
}


/*
 * IO Layers
 */
Layer* NewVectorLayer(unsigned int size) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = VECTOR_LAYER;
    layer->vectorLayer = malloc(sizeof(VectorLayer));
    layer->vectorLayer->size = size;

    layer->computedValues = NULL;

    return layer;
}
void FreeVectorLayer(Layer *layer) {
    free(layer->vectorLayer);
    free(layer);
}

Layer* NewMatrixLayer(unsigned int r, unsigned int c) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = MATRIX_LAYER;
    layer->matrixLayer = malloc(sizeof(MatrixLayer));
    layer->matrixLayer->r = r;
    layer->matrixLayer->c = c;

    layer->computedValues = NULL;

    return layer;
}
void FreeMatrixLayer(Layer *layer) {
    free(layer->matrixLayer);
    free(layer);
}


/*
 * Non-Trainable layers
 */
Layer* NewElementWiseLayer(double (*function)(double)) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = ELEMENTWISE_LAYER;
    layer->elementWiseLayer = malloc(sizeof(ElementWiseLayer));
    layer->elementWiseLayer->function = function;

    layer->computedValues = NULL;

    return layer;
}
void FreeElementWiseLayer(Layer *layer) {
    free(layer->elementWiseLayer);
    free(layer);
}

Layer* NewPoolingLayer(unsigned int poolSize, unsigned int poolingMethod) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = POOLING_LAYER;
    layer->poolingLayer = malloc(sizeof(PoolingLayer));
    layer->poolingLayer->poolSize = poolSize;
    layer->poolingLayer->poolingMethod = poolingMethod;

    layer->computedValues = NULL;

    return layer;
}

void FreePoolingLayer(Layer *layer) {
    free(layer->poolingLayer);
    free(layer);
}


/*
 * Non-Parameterised layers
 */
Layer* NewFlatteningLayer(void) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = FLATTENING_LAYER;
    layer->flatteningLayer = malloc(sizeof(FlatteningLayer));

    layer->computedValues = NULL;

    return layer;
}
void FreeFlatteningLayer(Layer *layer) {
    free(layer->flatteningLayer);
    free(layer);
}

Layer* NewSoftMaxLayer(void) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = SOFTMAX_LAYER;
    layer->softMaxLayer = malloc(sizeof(SoftMaxLayer));

    layer->computedValues = NULL;

    return layer;
}
void FreeSoftMaxLayer(Layer *layer) {
    free(layer->softMaxLayer);
    free(layer);
}


void FreeLayer(Layer *layer) {
    if (layer->computedValues != NULL) {
        FreeTensor(layer->computedValues);
    }

    switch (layer->uType) {
        case LINEAR_LAYER: {
            FreeLinearLayer(layer);
            break;
        }
        case CONVOLUTION_LAYER: {
            FreeConvolutionLayer(layer);
            break;
        }

        case VECTOR_LAYER: {
            FreeVectorLayer(layer);
            break;
        }
        case MATRIX_LAYER: {
            FreeMatrixLayer(layer);
            break;
        }

        case ELEMENTWISE_LAYER: {
            FreeElementWiseLayer(layer);
            break;
        }
        case POOLING_LAYER: {
            FreePoolingLayer(layer);
            break;
        }
        case FLATTENING_LAYER: {
            FreeFlatteningLayer(layer);
            break;
        }
        case SOFTMAX_LAYER: {
            FreeSoftMaxLayer(layer);
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in FreeLayer().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
} //FreeLayer()

void InitIOLayerTensor(Layer *layer) {
    switch (layer->uType) {
        case VECTOR_LAYER: {
            layer->computedValues = NewTensorVector(NewFilledVector(layer->vectorLayer->size, 0));
            break;
        }
        case MATRIX_LAYER: {
            layer->computedValues = NewTensorMatrix(NewFilledMatrix(layer->matrixLayer->r, layer->matrixLayer->c, 0));
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in InitIOLayerTensor().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
    return;

    error: {
        fprintf(stderr, "Error during IO Layer tensor initialisation");
        exit(EXIT_FAILURE_CODE);
    }
} //InitIOLayerTensor()

void InitHiddenLayerTensor(Layer *previousLayer, Layer *layer) {
    switch (layer->uType) {
        case LINEAR_LAYER: {
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }

            layer->computedValues = NewTensorVector(NewFilledVector(layer->linearLayer->size, 0));

            break;
        }
        case CONVOLUTION_LAYER: {

            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }
            unsigned int padding = layer->convolutionLayer->kernelSize - 1;
            unsigned int depth = 1;
            unsigned int r, c;
            if (previousLayer->computedValues->uType == MATRIX3D) {
                depth = previousLayer->computedValues->matrix3d->depth;
                r = previousLayer->computedValues->matrix3d->r;
                c = previousLayer->computedValues->matrix3d->c;
            } else {
                r = previousLayer->computedValues->matrix2d->r;
                c = previousLayer->computedValues->matrix2d->c;
            }

            layer->computedValues = NewTensorMatrix3d(NewFilledMatrix3d(depth * layer->convolutionLayer->kernelCount, r-padding, c-padding, 0));

            break;
        }

        case ELEMENTWISE_LAYER: {
            layer->computedValues = CloneTensorEmpty(previousLayer->computedValues);

            break;
        }
        case POOLING_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }

            const unsigned int poolSize = layer->poolingLayer->poolSize;

            switch (previousLayer->computedValues->uType) {
                case MATRIX2D: {
                    if (previousLayer->computedValues->matrix2d->r % poolSize != 0 || previousLayer->computedValues->matrix2d->c % poolSize != 0) {
                        fprintf(stderr, "Error: invalid pool size %d for matrix rxc. InitHiddenLayerTensor().", poolSize);
                        exit(EXIT_FAILURE_CODE);
                    }

                    layer->computedValues = NewTensorMatrix(NewFilledMatrix(previousLayer->computedValues->matrix2d->r / poolSize, previousLayer->computedValues->matrix2d->c / poolSize, 0));

                    break;
                }
                case MATRIX3D: {
                    if (previousLayer->computedValues->matrix3d->r % poolSize != 0 || previousLayer->computedValues->matrix3d->c % poolSize != 0) {
                        fprintf(stderr, "Error: invalid pool size %d for matrix rxc. InitHiddenLayerTensor().", layer->poolingLayer->poolSize);
                        exit(EXIT_FAILURE_CODE);
                    }

                    layer->computedValues =NewTensorMatrix3d(NewFilledMatrix3d(previousLayer->computedValues->matrix3d->depth, previousLayer->computedValues->matrix3d->r / poolSize, previousLayer->computedValues->matrix3d->c / poolSize, 0));

                    break;
                }
                default: {
                    fprintf(stderr, "Error: tried to access utype '%d' for a tensor InitHiddenLayerTensor().", previousLayer->computedValues->uType);
                    exit(EXIT_FAILURE_CODE);
                }
            }

            break;
        }

        case FLATTENING_LAYER: {
            layer->computedValues = NewTensorVector(NewFilledVector(previousLayer->computedValues->size, 0));

            break;
        }
        case SOFTMAX_LAYER: {
            layer->computedValues = CloneTensorEmpty(previousLayer->computedValues);

            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a layer in InitHiddenLayerTensor().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
    return;

    error: {
        fprintf(stderr, "Error: invalid layer sequence. Prev: %d, Current: %d", previousLayer->uType, layer->uType);
        exit(EXIT_FAILURE_CODE);
    }
} //InitHiddenLayerTensor()

// TO-DO
void ComputeLayerValues(Layer *previousLayer, Layer *layer) {
    switch (layer->uType) {
        case LINEAR_LAYER: {
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }

            Vector *v = VectorMatrixMultiply(layer->linearLayer->weights, previousLayer->computedValues->vector);

            if (v->size != layer->computedValues->size) {
                fprintf(stderr, "Error: weight matrix * previous layer vector size and current layer vector size do not match.");
                exit(EXIT_FAILURE_CODE);
            }

            AddVector(v, layer->linearLayer->biases);

            memcpy(GetTensorValues(layer->computedValues), v->values, sizeof(double)*v->size);

            FreeVector(v);

            break;
        }
        case CONVOLUTION_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }

            switch (previousLayer->computedValues->uType) {
                case MATRIX2D: {

                    for (unsigned int i = 0; i<layer->convolutionLayer->kernelCount; i++) {//Iterate over each kernel/filter

                        Matrix *mConvolved = ConvolveMatrix(previousLayer->computedValues->matrix2d, layer->convolutionLayer->kernels[i]);

                        Set2dSliceMatrix3d(layer->computedValues->matrix3d, mConvolved, i);

                        FreeMatrix(mConvolved);
                    }
                    break;
                }
                case MATRIX3D: {

                    for (unsigned int i = 0; i<layer->convolutionLayer->kernelCount; i++) {//Iterate over each kernel/filter


                        for (unsigned int j = 0; j<previousLayer->computedValues->matrix3d->depth; j++) {//Iterate each kernel/filter over each 2d matrix in the previous layer

                            Matrix *m = Get2dSliceMatrix3d(previousLayer->computedValues->matrix3d, j); // don't need to free

                            Matrix *mConvolved = ConvolveMatrix(m, layer->convolutionLayer->kernels[i]); // do need to free

                            Set2dSliceMatrix3d(layer->computedValues->matrix3d, mConvolved, i*layer->convolutionLayer->kernelCount + j);

                            FreeMatrix(mConvolved);
                        }
                    }
                    break;
                }
            }
            break;
        }

        case VECTOR_LAYER: {
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, true);

            break;
        }
        case MATRIX_LAYER: {
            if (previousLayer->computedValues->uType != MATRIX2D) {
                goto error;
            }

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, true);

            break;
        }

        case ELEMENTWISE_LAYER: {

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, true);

            double *values = GetTensorValues(layer->computedValues);

            for (int i = 0; i<layer->computedValues->size; i++) {
                values[i] = layer->elementWiseLayer->function(values[i]);
            }

            break;
        }
        case POOLING_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }

            const unsigned int poolSize = layer->poolingLayer->poolSize;

            switch (previousLayer->computedValues->uType) {
                case MATRIX2D: {

                    switch (layer->poolingLayer->poolingMethod) {
                        case MAX_POOLING: {
                            MaxPoolMatrix(layer->computedValues->matrix2d, previousLayer->computedValues->matrix2d, poolSize);
                            break;
                        }
                        case AVERAGE_POOLING: {
                            AveragePoolMatrix(layer->computedValues->matrix2d, previousLayer->computedValues->matrix2d, poolSize);
                            break;
                        }
                    }
                    break;
                }
                case MATRIX3D: {

                    switch (layer->poolingLayer->poolingMethod) {
                        case MAX_POOLING: {
                            MaxPoolMatrix3d(layer->computedValues->matrix3d, previousLayer->computedValues->matrix3d, poolSize);
                            break;
                        }
                        case AVERAGE_POOLING: {
                            AveragePoolMatrix3d(layer->computedValues->matrix3d, previousLayer->computedValues->matrix3d, poolSize);
                            break;
                        }
                    }
                    break;
                }
            }
            break;
        }
        case FLATTENING_LAYER: {

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, false);

            break;
        }
        case SOFTMAX_LAYER: {

            double* previousValues = GetTensorValues(previousLayer->computedValues);
            double* values = GetTensorValues(previousLayer->computedValues);

            double sum = 0;

            for (int i = 0; i<previousLayer->computedValues->size; i++) {
                sum += exp(previousValues[i]);
            }

            for (int i = 0; i<previousLayer->computedValues->size; i++) {
                values[i] = exp(previousValues[i]) / sum;
            }

            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in ComputeLayerValues().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
    return;

    error: {
        fprintf(stderr, "Error: invalid layer sequence. Prev: %d, Current: %d", previousLayer->uType, layer->uType);
        exit(EXIT_FAILURE_CODE);
    }
} // InitHiddenLayerTensor()

NeuralNetwork* NewNeuralNetwork(unsigned int maxTrainingRounds, double learningRate) {
    NeuralNetwork *nNet = malloc(sizeof(NeuralNetwork));

    nNet->stage = NET_EMPTY;
    nNet->hiddenLayerCount = 0;
    nNet->maxTrainingRounds = maxTrainingRounds;
    nNet->learningRate = learningRate;

    nNet->inputLayer = NULL;
    nNet->outputLayer = NULL;
    nNet->hiddenLayers = NULL;

    return nNet;
}
void FreeNeuralNetwork(NeuralNetwork *nNet) {
    FreeLayer(nNet->inputLayer);
    FreeLayer(nNet->outputLayer);
    for (unsigned int i = 0; i<nNet->hiddenLayerCount; i++) {
        FreeLayer(nNet->hiddenLayers[i]);
    }
    free(nNet);
}

//precondition: Input layer is initialised
void AddHiddenLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > 1) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = 1;

    nNet->hiddenLayerCount++;
    Layer** temp = malloc(sizeof(Layer**) * nNet->hiddenLayerCount);
    memcpy(temp, nNet->hiddenLayers, sizeof(Layer**) * (nNet->hiddenLayerCount-1));
    nNet->hiddenLayers = temp;
    nNet->hiddenLayers[nNet->hiddenLayerCount-1] = layer;
}

void SetInputLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > 1) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = 1;
    nNet->inputLayer = layer;
}

//precondition: Hidden layers are finalised
void SetOutputLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > 1) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = 1;
    nNet->outputLayer = layer;
}

void FinalizeNeuralNetworkLayers(NeuralNetwork *nNet) {
    if (nNet->stage != 1) {
        fprintf(stderr, "Error: Attempted to finalize layers of network in stage %d", nNet->stage);
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = 2;

    InitIOLayerTensor(nNet->inputLayer);

    // The first hidden layer's tensors are initialised outside the for loop as the previous layer will be the input layer;
    InitHiddenLayerTensor(nNet->inputLayer, nNet->hiddenLayers[0]);

    for (int i = 1; i<nNet->hiddenLayerCount; i++) {
        InitHiddenLayerTensor(nNet->hiddenLayers[i-1], nNet->hiddenLayers[i]);
        if (nNet->hiddenLayers[i]->uType == LINEAR_LAYER) {
            nNet->hiddenLayers[i]->linearLayer->weights = NewRandomisedMatrix(nNet->hiddenLayers[i]->computedValues->size, nNet->hiddenLayers[i-1]->computedValues->size);
        }
    }

    InitIOLayerTensor(nNet->outputLayer);
} //Call after adding layers

Tensor* RunNeuralNetwork(NeuralNetwork *nNet, Tensor *input) {
    if (nNet->stage < 3) {
        fprintf(stderr, "Error: Attempted to run untrained neural network (stage %d). Required stage: >= 3", nNet->stage);
        //exit(EXIT_FAILURE_CODE);
    }

    CopyTensorValues(nNet->inputLayer->computedValues, input, true);

    // The first hidden layer's tensors are calculated outside the for loop as the previous layer will be the input layer;
    ComputeLayerValues(nNet->inputLayer, nNet->hiddenLayers[0]);

    for (int i = 1; i<nNet->hiddenLayerCount; i++) {
        ComputeLayerValues(nNet->hiddenLayers[i-1], nNet->hiddenLayers[i]);
    }

    ComputeLayerValues(nNet->hiddenLayers[nNet->hiddenLayerCount-1], nNet->outputLayer);

    return nNet->outputLayer->computedValues;
}

int NeuralNetworkMain() {

    NeuralNetwork *nNet = NewNeuralNetwork(6, 0.05);

    SetInputLayer(nNet, NewMatrixLayer(28, 28));
    AddHiddenLayer(nNet, NewConvolutionLayer(8, 3));
    AddHiddenLayer(nNet, NewPoolingLayer(2, MAX_POOLING));
    AddHiddenLayer(nNet, NewFlatteningLayer());
    AddHiddenLayer(nNet, NewLinearLayer(500));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu));
    AddHiddenLayer(nNet, NewLinearLayer(100));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu));
    AddHiddenLayer(nNet, NewLinearLayer(10));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu));
    AddHiddenLayer(nNet, NewSoftMaxLayer());
    SetOutputLayer(nNet, NewVectorLayer(10));

    FinalizeNeuralNetworkLayers(nNet);

    // CSV-MNIST

    CSVFile *csv;

    // Windows
    csv = OpenCSVFile("..\\data\\mnist_test.csv");

    // Unix
    // csv = OpenCSVFile("../data/mnist_test.csv");

    CSVInfo(csv);

    int count = 0;
    char buffer[CSV_LINE_MAX_BUFF];

    SkipLine(csv);

    MnistDigit *mnistDigit = NewMnistDigit();

    for (int i = 0; i<1; i++) {
        ReadDigitFromCSV(csv, mnistDigit);
        PrintMnistDigit(mnistDigit);
        Tensor* inputTensor = NewTensorMatrix(mnistDigit->pixels);
        Tensor *netOutput = RunNeuralNetwork(nNet, inputTensor);
        FreeTensor(inputTensor);



        PrintVectorHorizontal(netOutput->vector);
    }


    //Cleanup
    FreeNeuralNetwork(nNet);
    free(mnistDigit);
    FreeCSV(csv);

    return 0;
}
