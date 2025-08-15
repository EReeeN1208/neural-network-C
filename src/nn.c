//
// Created by erenk on 8/13/2025.
//

#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

#include "activationFunctions.h"
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

            layer->computedValues = NewTensorMatrix3d(NewFilledMatrix3d(depth * layer->convolutionLayer->kernelCount, r, c, 0));

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
                    if (previousLayer->computedValues->matrix2d->r % poolSize != 0 || previousLayer->computedValues->matrix2d->r % poolSize != 0) {
                        fprintf(stderr, "Error: invalid pool size %d for matrix rxc. InitHiddenLayerTensor().", poolSize);
                        exit(EXIT_FAILURE_CODE);
                    }

                    layer->computedValues = NewTensorMatrix(NewFilledMatrix(previousLayer->computedValues->matrix2d->r / poolSize, previousLayer->computedValues->matrix2d->c / poolSize, 0));

                    break;
                }
                case MATRIX3D: {
                    if (previousLayer->computedValues->matrix3d->r % poolSize != 0 || previousLayer->computedValues->matrix3d->r % poolSize != 0) {
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
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in InitHiddenLayerTensor().", layer->uType);
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

            break;
        }
        case CONVOLUTION_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }

            break;
        }

        case VECTOR_LAYER: {
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }

            break;
        }
        case MATRIX_LAYER: {
            if (previousLayer->computedValues->uType != MATRIX2D) {
                goto error;
            }

            break;
        }

        case ELEMENTWISE_LAYER: {

            break;
        }
        case POOLING_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }

            break;
        }
        case FLATTENING_LAYER: {

            break;
        }
        case SOFTMAX_LAYER: {

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
    /*
    if (nNet->inputLayer == NULL) {
        fprintf(stderr, "Error: Attempted to add hidden layer before initialising the network's input layer.");
        exit(EXIT_FAILURE_CODE);
    }
    if (nNet->outputLayer != NULL) {
        fprintf(stderr, "Error: Attempted to add hidden layer after initialising the network's output layer.");
        exit(EXIT_FAILURE_CODE);
    }
    */
    if (nNet->stage > 1) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = 1;

    nNet->hiddenLayerCount++;
    realloc(nNet->hiddenLayers, sizeof(Layer) * nNet->hiddenLayerCount);
    nNet->hiddenLayers[nNet->hiddenLayerCount - 1] = layer;
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
    }

    InitIOLayerTensor(nNet->outputLayer);

    InitNetworkWeights(nNet);
} //Call after adding layers

void InitNetworkWeights(NeuralNetwork *nNet) {
    // TO-DO
}

int NeuralNetworkMain() {

    NeuralNetwork *nNet = NewNeuralNetwork(6, 0.05);
    SetInputLayer(nNet, NewMatrixLayer(28, 28));
    SetOutputLayer(nNet, NewVectorLayer(10));


    //Cleanup
    FreeNeuralNetwork(nNet);

    return 0;
}
