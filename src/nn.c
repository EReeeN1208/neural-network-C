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
    layer->linearLayer->weightGradients = NULL; // will be allocated during InitNeuralNetworkLayers()
    layer->linearLayer->biases = NewFilledVector(size, 0);
    layer->linearLayer->biasGradients = NewFilledVector(size, 0);

    layer->computedValues = NULL;
    layer->computedValueGradients = NULL;

    return layer;
}
void FreeLinearLayer(Layer *layer) {
    FreeMatrix(layer->linearLayer->weights);
    FreeMatrix(layer->linearLayer->weightGradients);
    FreeVector(layer->linearLayer->biases);
    FreeVector(layer->linearLayer->biasGradients);
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
    layer->convolutionLayer->kernelGradients = malloc(sizeof(Matrix*) * kernelCount);
    layer->convolutionLayer->kernelBiases = NewFilledVector(kernelCount, 0);
    for (int i = 0; i<kernelCount; i++) {
        layer->convolutionLayer->kernelGradients[i] = NewFilledMatrix(kernelSize, kernelSize, 0);
        layer->convolutionLayer->kernels[i] = NewHeRandomMatrix(kernelSize, kernelSize);
        /*
        switch (i % 4) {
            case 0: {
                layer->convolutionLayer->kernels[i] = NewHeRandomMatrix(kernelSize, kernelSize);
                break;
            }
            case 1: {
                layer->convolutionLayer->kernels[i] = NewHeRandomMatrix(kernelSize, kernelSize);
                AddDoubleToMatrix(layer->convolutionLayer->kernels[i], 0.5);
                break;
            }
            case 2: {
                layer->convolutionLayer->kernels[i] = NewHeRandomMatrix(kernelSize, kernelSize);
                AddDoubleToMatrix(layer->convolutionLayer->kernels[i], 1);
                break;
            }
            case 3: {
                layer->convolutionLayer->kernels[i] = NewFilledMatrix(kernelSize, kernelSize, 1);
                break;
            }
        }
        */
    }

    layer->computedValues = NULL;
    layer->computedValueGradients = NULL;

    return layer;
}
void FreeConvolutionLayer(Layer *layer) {
    for (int i = 0; i<layer->convolutionLayer->kernelCount; i++) {
        FreeMatrix(layer->convolutionLayer->kernels[i]);
        FreeMatrix(layer->convolutionLayer->kernelGradients[i]);
    }
    FreeVector(layer->convolutionLayer->kernelBiases);
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
    layer->computedValueGradients = NULL;

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
    layer->computedValueGradients = NULL;

    return layer;
}
void FreeMatrixLayer(Layer *layer) {
    free(layer->matrixLayer);
    free(layer);
}


/*
 * Non-Trainable layers
 */
Layer* NewElementWiseLayer(double (*function)(double), double (*gradientFunction)(double)) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = ELEMENTWISE_LAYER;
    layer->elementWiseLayer = malloc(sizeof(ElementWiseLayer));
    layer->elementWiseLayer->function = function;
    layer->elementWiseLayer->gradientFunction = gradientFunction;

    layer->computedValues = NULL;
    layer->computedValueGradients = NULL;

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
    layer->computedValueGradients = NULL;

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
    layer->computedValueGradients = NULL;

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
    layer->computedValueGradients = NULL;

    return layer;
}
void FreeSoftMaxLayer(Layer *layer) {
    free(layer->softMaxLayer);
    free(layer);
}

Layer* NewReshapeLayer(void) {
    Layer *layer = malloc(sizeof(Layer));
    layer->uType = RESHAPE_LAYER;
    layer->reshapeLayer = malloc(sizeof(ReshapeLayer));

    layer->computedValues = NULL;
    layer->computedValueGradients = NULL;

    return layer;
}
void FreeReshapeLayer(Layer *layer) {
    free(layer->reshapeLayer);
    free(layer);
}


void FreeLayer(Layer *layer) {
    if (!layer) {
        return;
    }

    if (layer->computedValues) {
        FreeTensor(layer->computedValues);
    }
    if (layer->computedValueGradients) {
        FreeTensor(layer->computedValueGradients);
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
        case RESHAPE_LAYER: {
            FreeReshapeLayer(layer);
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a layer in FreeLayer().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
} //FreeLayer()

void PrintLayerInfo(Layer *layer) {
    char layerName[LAYER_WRITE_MAX_BUFF];
    char tensorInfo[TENSOR_WRITE_MAX_BUFF];

    switch (layer->uType) {
        case LINEAR_LAYER: {
            strcpy(layerName, "Linear Layer");
            break;
        }
        case CONVOLUTION_LAYER: {
            strcpy(layerName, "Convolution Layer");
            break;
        }

        case VECTOR_LAYER: {
            strcpy(layerName, "Vector Layer");
            break;
        }
        case MATRIX_LAYER: {
            strcpy(layerName, "Matrix Layer");
            break;
        }

        case ELEMENTWISE_LAYER: {
            strcpy(layerName, "Elementwise Layer");
            break;
        }
        case POOLING_LAYER: {
            strcpy(layerName, "Pooling Layer");
            break;
        }
        case FLATTENING_LAYER: {
            strcpy(layerName, "Flattening Layer");
            break;
        }
        case SOFTMAX_LAYER: {
            strcpy(layerName, "Softmax Layer");
            break;
        }
        case RESHAPE_LAYER: {
            strcpy(layerName, "Reshape Layer");
            break;
        }
        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a layer in PrintLayerInfo().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }

    WriteTensorInfo(layer->computedValues, tensorInfo);
    printf("%-30s | %s\n", layerName, tensorInfo);
}

void InitIOLayerTensor(Layer *layer) {
    switch (layer->uType) {
        case VECTOR_LAYER: {
            layer->computedValues = NewTensorEncapsulateVector(NewFilledVector(layer->vectorLayer->size, 0));
            break;
        }
        case MATRIX_LAYER: {
            layer->computedValues = NewTensorEncapsulateMatrix(NewFilledMatrix(layer->matrixLayer->r, layer->matrixLayer->c, 0));
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a tensor in InitIOLayerTensor().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }

    layer->computedValueGradients = CloneTensorEmpty(layer->computedValues);
} //InitIOLayerTensor()

void InitHiddenLayerTensor(Layer *previousLayer, Layer *layer) {
    switch (layer->uType) {
        case LINEAR_LAYER: {
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }

            layer->computedValues = NewTensorEncapsulateVector(NewFilledVector(layer->linearLayer->size, 0));

            break;
        }
        case CONVOLUTION_LAYER: {
            if (previousLayer->computedValues->uType != MATRIX3D) {
                goto error;
            }

            unsigned int padding = layer->convolutionLayer->kernelSize - 1;
            unsigned int depth = previousLayer->computedValues->matrix3d->depth;
            unsigned int r = previousLayer->computedValues->matrix3d->r;
            unsigned int c = previousLayer->computedValues->matrix3d->c;

            layer->computedValues = NewTensorEncapsulateMatrix3d(NewFilledMatrix3d(depth * layer->convolutionLayer->kernelCount, r-padding, c-padding, 0));

            break;
        }

        case ELEMENTWISE_LAYER: {
            layer->computedValues = CloneTensorEmpty(previousLayer->computedValues);

            break;
        }
        case POOLING_LAYER: {
            if (previousLayer->computedValues->uType != MATRIX3D) {
                goto error;
            }

            const unsigned int poolSize = layer->poolingLayer->poolSize;

            if (previousLayer->computedValues->matrix3d->r % poolSize != 0 || previousLayer->computedValues->matrix3d->c % poolSize != 0) {
                fprintf(stderr, "Error: invalid pool size %d for matrix rxc. InitHiddenLayerTensor().", layer->poolingLayer->poolSize);
                exit(EXIT_FAILURE_CODE);
            }

            layer->computedValues = NewTensorEncapsulateMatrix3d(NewFilledMatrix3d(previousLayer->computedValues->matrix3d->depth, previousLayer->computedValues->matrix3d->r / poolSize, previousLayer->computedValues->matrix3d->c / poolSize, 0));

            break;
        }

        case FLATTENING_LAYER: {
            if (previousLayer->computedValues->uType == VECTOR) {
                goto error;
            }
            layer->computedValues = NewTensorEncapsulateVector(NewFilledVector(previousLayer->computedValues->size, 0));

            break;
        }
        case SOFTMAX_LAYER: {
            layer->computedValues = CloneTensorEmpty(previousLayer->computedValues);

            break;
        }
        case RESHAPE_LAYER: {
            // uncomment to allow support for 3d matrices to also be reshaped back into 3d matrices
            /*
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

            Matrix3d *m3d = NewEmptyMatrix3d(depth, r, c);

            layer->computedValues = NewTensorEncapsulateMatrix3d(m3d);

            break;
            */
            if (previousLayer->computedValues->uType != MATRIX2D) {
                goto error;
            }
            Matrix3d *m3d = NewEmptyMatrix3d(1, previousLayer->computedValues->matrix2d->r, previousLayer->computedValues->matrix2d->c);

            layer->computedValues = NewTensorEncapsulateMatrix3d(m3d);
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a layer in InitHiddenLayerTensor()", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }

    layer->computedValueGradients = CloneTensorEmpty(layer->computedValues);

    return;

    error: {
        if (layer->uType == CONVOLUTION_LAYER || layer->uType == POOLING_LAYER) {
            fprintf(stderr, "Error: Convolution and pooling layers only support layers with 3d matrices as inputs.\nUse a reshape layer to convert a 2d layer into a 3d layer. InitHiddenLayerTensor()");
        } else {
            fprintf(stderr, "Error: invalid layer sequence. Prev: %d, Current: %d", previousLayer->uType, layer->uType);
        }
        exit(EXIT_FAILURE_CODE);
    }
} //InitHiddenLayerTensor()


void ComputeLayerValues(Layer *previousLayer, Layer *layer) {
    switch (layer->uType) {
        case LINEAR_LAYER: {
            /* Should already be checked during tensor init
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }
            */

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
            /* Should already be checked during tensor init
            if (previousLayer->computedValues->uType != MATRIX3D) {
                goto error;
            }
            */

            for (unsigned int i = 0; i<layer->convolutionLayer->kernelCount; i++) {//Iterate over each kernel/filter

                for (unsigned int j = 0; j<previousLayer->computedValues->matrix3d->depth; j++) {//Iterate each kernel/filter over each 2d matrix in the previous layer

                    Matrix *m = Get2dSliceMatrix3d(previousLayer->computedValues->matrix3d, j); // don't need to free

                    Matrix *mConvolved = ConvolveMatrix(m, layer->convolutionLayer->kernels[i]); // do need to free

                    AddDoubleToMatrix(mConvolved, GetVectorValue(layer->convolutionLayer->kernelBiases, i));

                    Set2dSliceMatrix3d(layer->computedValues->matrix3d, mConvolved, i*previousLayer->computedValues->matrix3d->depth + j);

                    FreeMatrix(mConvolved);
                }
            }
            break;
        }

        case VECTOR_LAYER: {
            /* Should already be checked during tensor init
            if (previousLayer->computedValues->uType != VECTOR) {
                goto error;
            }
            */

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, true);

            break;
        }
        case MATRIX_LAYER: {
            /* Should already be checked during tensor init
            if (previousLayer->computedValues->uType != MATRIX2D) {
                goto error;
            }
            */

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
            /* Should already be checked during tensor init
            if (previousLayer->computedValues->uType == MATRIX3D) {
                goto error;
            }
            */

            const unsigned int poolSize = layer->poolingLayer->poolSize;

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
        case FLATTENING_LAYER: {

            CopyTensorValues(layer->computedValues, previousLayer->computedValues, false);

            break;
        }
        case SOFTMAX_LAYER: {

            double* previousValues = GetTensorValues(previousLayer->computedValues);
            double* values = GetTensorValues(layer->computedValues);

            double max = previousValues[0];

            for (int i = 1; i<previousLayer->computedValues->size; i++) {
                if (previousValues[i] > max) {
                    max = previousValues[i];
                }
            }

            double sum = 0;

            for (int i = 0; i<previousLayer->computedValues->size; i++) {
                values[i] = exp(previousValues[i] - max);
                sum += values[i];
            }

            for (int i = 0; i<previousLayer->computedValues->size; i++) {
                values[i] /= sum;
            }

            break;
        }

        case RESHAPE_LAYER: {
            Set2dSliceMatrix3d(layer->computedValues->matrix3d, previousLayer->computedValues->matrix2d, 0);
            break;
        }

        default: {
            fprintf(stderr, "Error: tried to access utype '%d' for a layer in ComputeLayerValues().", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
    /*
    return;

    error: {
        fprintf(stderr, "Error: invalid layer sequence. Prev: %d, Current: %d", previousLayer->uType, layer->uType);
        exit(EXIT_FAILURE_CODE);
    }
    */
} // InitHiddenLayerTensor()

void ComputeHiddenLayerGradients(Layer *previousLayer, Layer *layer, Layer *proceedingLayer, double lr) {

    switch (layer->uType) {
        case LINEAR_LAYER: {
            CopyTensorValues(layer->computedValueGradients, proceedingLayer->computedValueGradients, false);


            Matrix* weightGradients = layer->linearLayer->weightGradients;

            for (int i = 0; i < weightGradients->r; i++) {
                for (int j = 0; j < weightGradients->c; j++) {
                    double gradient = GetTensorValues(layer->computedValueGradients)[i] *GetTensorValues(previousLayer->computedValues)[j];
                    SetMatrixValueRowCol(weightGradients, i, j, gradient);
                }
                layer->linearLayer->biasGradients->values[i] = GetTensorValues(layer->computedValueGradients)[i];
            }

            Matrix *weightsTransposed = TransposeMatrix(layer->linearLayer->weights);
            Vector *vGradients = VectorMatrixMultiply(weightsTransposed, layer->computedValueGradients->vector);

            for (int i = 0; i < vGradients->size; i++) {
                GetTensorValues(previousLayer->computedValueGradients)[i] = vGradients->values[i];
            }

            FreeVector(vGradients);
            FreeMatrix(weightsTransposed);

            for (int i = 0; i < weightGradients->r; i++) {
                for (int j = 0; j < weightGradients->c; j++) {
                    double gradient = GetMatrixValueRowCol(weightGradients, i, j);
                    double weight = GetMatrixValueRowCol(layer->linearLayer->weights, i, j);
                    SetMatrixValueRowCol(layer->linearLayer->weights, i, j, weight - lr*gradient);
                }
                layer->linearLayer->biases->values[i] -= lr * layer->linearLayer->biasGradients->values[i];
            }

            break;
        }
        case CONVOLUTION_LAYER: {



            //this block is commented the hell out of because it was really complex the program. I left them in because maybe they will be helpful for someone

            unsigned int kernelSize = layer->convolutionLayer->kernelSize;
            unsigned int kernelCount = layer->convolutionLayer->kernelCount;

            /*
            printf("\n\n\n=====start=====\n");

            printf("old kernelBiases vector:");
            PrintVectorHorizontal(layer->convolutionLayer->kernelBiases);

            printf("\n\nold kernelGradients kernel 0");
            PrintMatrix(layer->convolutionLayer->kernelGradients[0]);
            */

            for (int i = 0; i<kernelCount; i++) {//iterate over each kernel
                Matrix *mWeightGradients = layer->convolutionLayer->kernelGradients[i]; //weight gradients of current kernel
                ZeroValues(mWeightGradients->values, mWeightGradients->r * mWeightGradients->c);

                double biasGradient = 0;

                for (int r = 0; r<layer->computedValues->matrix3d->r; r++) {
                    for (int c = 0; c<layer->computedValues->matrix3d->c; c++) {// these 2 loops iterate over each r,c pos of the output 3d matrix.
                        for (int d = 0; d<previousLayer->computedValues->matrix3d->depth/*input matrix #*/; d++) { //this loop iterates the (r,c) over depth to access each (r,c) in the input matrice
                            double upstreamGradient = GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, i*previousLayer->computedValues->matrix3d->depth + d, r, c);
                            for (int j = 0; j<kernelSize*kernelSize; j++) { //this loop iterates over each kernel position
                                double inputValue = GetMatrix3DValueDepthRowCol(previousLayer->computedValues->matrix3d, d, r+(j/kernelSize), c+(j%kernelSize));
                                mWeightGradients->values[j] += inputValue * upstreamGradient;
                            }
                            biasGradient += upstreamGradient;
                        }
                    }
                }
                layer->convolutionLayer->kernelBiases->values[i] -= biasGradient*lr;
            }

            /*
            printf("\n\nnew kernelBiases vector:");
            PrintVectorHorizontal(layer->convolutionLayer->kernelBiases);

            printf("\n\nnew kernelGradients kernel 0");
            PrintMatrix(layer->convolutionLayer->kernelGradients[0]);

            printf("\n\nold computedValueGradients slice 0");
            PrintMatrix3dSlice(previousLayer->computedValueGradients->matrix3d, 0);
            */

            ZeroTensorValues(previousLayer->computedValueGradients);

            //iterate over each output pixel
            for (int d = 0; d<layer->computedValues->matrix3d->depth; d++) {
                for (int r = 0; r<layer->computedValues->matrix3d->r; r++) {
                    for (int c = 0; c<layer->computedValues->matrix3d->c; c++) {
                        //iterate over each output pixel. For each output pixel, we need to multiply:
                        //the kernel weight that touched every input pixel that affected the output pixel and the upstream gradient for that output pixel.
                        //add that to the gradient of that input pixel

                        //upstream gradient at that output pixel.
                        double upstreamGradient = GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, d, r, c);

                        /*
                        for (int i = 0; i<layer->convolutionLayer->kernelCount; i++) {
                            for (int j = 0; j<kernelSize*kernelSize; j++) {
                                //iterating over each
                                GetMatrix3DValueDepthRowCol(previousLayer->computedValues->matrix3d, i*layer->convolutionLayer->kernelCount + d, r+(j/kernelSize), c+(j%kernelSize));
                            }
                        }*/

                        //for each output pixel, we need find the kernel that affected that layer, and then go b
                        unsigned int i = d/kernelCount; //i is the index of kernel used for this output layer
                        for (int j = 0; j<kernelSize*kernelSize; j++) {//now in this loop, we can iterate over pixels in the input layer like a convolution filter. kernelSize^2 pixels for each loop run.
                            //iterating over conv filter
                            AddDoubleToMatrix3dDepthRowCol(previousLayer->computedValueGradients->matrix3d, d/kernelCount, r+(j/kernelSize), c+(j%kernelSize), upstreamGradient * GetMatrixValuePos(layer->convolutionLayer->kernels[i], j));
                        }
                    }
                }
            }
            /*
            printf("\n\nnew computedValueGradients slice 0");
            PrintMatrix3dSlice(previousLayer->computedValueGradients->matrix3d, 0);

            printf("\n\nold kernel: kernel 0");
            PrintMatrix(layer->convolutionLayer->kernels[0]);
            */
            for (int i = 0; i<kernelCount; i++) { //update weights at the end
                Matrix *mWeightGradients = layer->convolutionLayer->kernelGradients[i];
                Matrix *mWeights = layer->convolutionLayer->kernels[i];

                ScaleMatrixDouble(mWeightGradients, lr);
                AddMatrix(mWeights, mWeightGradients);
            }
            /*
            printf("\n\nnew kernel: kernel 0");
            PrintMatrix(layer->convolutionLayer->kernels[0]);

            printf("\n=====end=====\n\n");
            */
            break;
        }
        case ELEMENTWISE_LAYER: {

            if (layer->computedValueGradients->size != previousLayer->computedValueGradients->size) {
                printf("amogus");
            }

            for (int i = 0; i < layer->computedValueGradients->size; i++) {
                double gradient = GetTensorValues(layer->computedValueGradients)[i] * layer->elementWiseLayer->gradientFunction(GetTensorValues(previousLayer->computedValues)[i]);
                GetTensorValues(previousLayer->computedValueGradients)[i] = gradient;
            }
            break;
        }
        case POOLING_LAYER: {
            const unsigned int poolSize = layer->poolingLayer->poolSize;
            const double poolSizeSquared = poolSize * poolSize;
            switch (layer->poolingLayer->poolingMethod) {
                case MAX_POOLING: {
                    ZeroTensorValues(previousLayer->computedValueGradients);
                    for (int d = 0; d<layer->computedValues->matrix3d->depth; d++) {
                        for (int i = 0; i<layer->computedValues->matrix3d->r; i++) {
                            for (int j = 0; j<layer->computedValues->matrix3d->c; j++) {
                                double max = GetMatrix3DValueDepthRowCol(layer->computedValues->matrix3d, d, i, j);
                                for (int r = 0; r<poolSize; r++) {
                                    for (int c = 0; c<poolSize; c++) {
                                        if (GetMatrix3DValueDepthRowCol(previousLayer->computedValues->matrix3d, d, i*poolSize + r, j*poolSize+c) == max) {
                                            SetMatrix3DValueDepthRowCol(previousLayer->computedValueGradients->matrix3d, d, i*poolSize + r, j*poolSize+c, GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, d, i, j));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                case AVERAGE_POOLING: {
                    for (int d = 0; d<previousLayer->computedValueGradients->matrix3d->depth; d++) {
                        for (int i = 0; i<previousLayer->computedValueGradients->matrix3d->r; i++) {
                            for (int j = 0; j<previousLayer->computedValueGradients->matrix3d->c; j++) {
                                SetMatrix3DValueDepthRowCol(previousLayer->computedValueGradients->matrix3d, d, i, j, GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, d, i/poolSize, j/poolSize) / poolSizeSquared);
                            }
                        }
                    }
                    break;
                }
            }
            break;

            /*
            switch (previousLayer->computedValues->uType) {
                case MATRIX2D: {

                    switch (layer->poolingLayer->poolingMethod) {

                        case MAX_POOLING: {
                            ZeroTensorValues(previousLayer->computedValueGradients);
                            for (int i = 0; i<layer->computedValues->matrix2d->r; i++) {
                                for (int j = 0; j<layer->computedValues->matrix2d->c; j++) {
                                    double max = GetMatrixValueRowCol(layer->computedValues->matrix2d, i, j);
                                    for (int r = 0; r<poolSize; r++) {
                                        for (int c = 0; c<poolSize; c++) {
                                            if (GetMatrixValueRowCol(previousLayer->computedValues->matrix2d, i*poolSize + r, j*poolSize+c) == max) {
                                                SetMatrixValueRowCol(previousLayer->computedValueGradients->matrix2d, i*poolSize + r, j*poolSize+c, GetMatrixValueRowCol(layer->computedValueGradients->matrix2d, i, j));
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case AVERAGE_POOLING: {
                            for (int i = 0; i<previousLayer->computedValueGradients->matrix2d->r; i++) {
                                for (int j = 0; j<previousLayer->computedValueGradients->matrix2d->c; j++) {
                                    SetMatrixValueRowCol(previousLayer->computedValueGradients->matrix2d, i, j, GetMatrixValueRowCol(layer->computedValueGradients->matrix2d, i/poolSize, j/poolSize) / poolSizeSquared);
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
                case MATRIX3D: {

                    switch (layer->poolingLayer->poolingMethod) {
                        case MAX_POOLING: {
                            ZeroTensorValues(previousLayer->computedValueGradients);
                            for (int d = 0; d<layer->computedValues->matrix3d->depth; d++) {
                                for (int i = 0; i<layer->computedValues->matrix3d->r; i++) {
                                    for (int j = 0; j<layer->computedValues->matrix3d->c; j++) {
                                        double max = GetMatrix3DValueDepthRowCol(layer->computedValues->matrix3d, d, i, j);
                                        for (int r = 0; r<poolSize; r++) {
                                            for (int c = 0; c<poolSize; c++) {
                                                if (GetMatrix3DValueDepthRowCol(previousLayer->computedValues->matrix3d, d, i*poolSize + r, j*poolSize+c) == max) {
                                                    SetMatrix3DValueDepthRowCol(previousLayer->computedValueGradients->matrix3d, d, i*poolSize + r, j*poolSize+c, GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, d, i, j));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case AVERAGE_POOLING: {
                            for (int d = 0; d<previousLayer->computedValueGradients->matrix3d->depth; d++) {
                                for (int i = 0; i<previousLayer->computedValueGradients->matrix3d->r; i++) {
                                    for (int j = 0; j<previousLayer->computedValueGradients->matrix3d->c; j++) {
                                        SetMatrix3DValueDepthRowCol(previousLayer->computedValueGradients->matrix3d, d, i, j, GetMatrix3DValueDepthRowCol(layer->computedValueGradients->matrix3d, d, i/poolSize, j/poolSize) / poolSizeSquared);
                                    }
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
            }
            */
            break;
        }
        case FLATTENING_LAYER: {

            CopyTensorValues(previousLayer->computedValueGradients, layer->computedValueGradients, false);
            break;
        }
        case SOFTMAX_LAYER: {

            CopyTensorValues(previousLayer->computedValueGradients, layer->computedValueGradients, false);
            break;
        }
        case RESHAPE_LAYER: {
            CopyTensorValues(previousLayer->computedValueGradients, layer->computedValueGradients, false);
            break;
        }

        default: {
            fprintf(stderr, "Error: Unknown layer type '%d' in ComputeLayerGradients()\n", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
}

void ComputeIOLayerGradients(Layer *previousLayer, Layer *layer) {
    switch (layer->uType) {
        case VECTOR_LAYER: {

            CopyTensorValues(previousLayer->computedValueGradients, layer->computedValueGradients, false);
            break;
        }
        case MATRIX_LAYER: {

            CopyTensorValues(previousLayer->computedValueGradients, layer->computedValueGradients, false);
            break;
        }
        default: {
            fprintf(stderr, "Error: Unknown layer type '%d' in ComputeIOLayerGradients()\n", layer->uType);
            exit(EXIT_FAILURE_CODE);
        }
    }
}

NeuralNetwork* NewNeuralNetwork(unsigned int trainingRounds, unsigned int maxRoundSteps, double learningRate, char *name) {
    NeuralNetwork *nNet = malloc(sizeof(NeuralNetwork));

    nNet->name = malloc(sizeof(char) * (strlen(name)+1));

    memcpy(nNet->name, name, strlen(name) + 1);

    nNet->stage = NET_EMPTY;
    nNet->hiddenLayerCount = 0;
    nNet->trainingRounds = trainingRounds;
    nNet->maxRoundSteps = maxRoundSteps;
    nNet->learningRate = learningRate;

    nNet->inputLayer = NULL;
    nNet->outputLayer = NULL;
    nNet->hiddenLayers = NULL;

    nNet->trainingStepCount = 0;

    return nNet;
}

void PrintNeuralNetworkInfo(NeuralNetwork *nNet) {
    printf("\nNeural Network '%s' info:\n", nNet->name);
    printf("Stage: %d, Layer count: %d, Training Rounds: %d, Training Rounds Max Steps: %d, Learning Rate: %f, Training Steps Count: %d\n", nNet->stage, nNet->hiddenLayerCount+2, nNet->trainingRounds, nNet->maxRoundSteps, nNet->learningRate, nNet->trainingStepCount);
    printf("Network Layers:\n");
    PrintLayerInfo(nNet->inputLayer);
    for (int i = 0; i<nNet->hiddenLayerCount; i++) {
        PrintLayerInfo(nNet->hiddenLayers[i]);
    }
    PrintLayerInfo(nNet->outputLayer);
}

void FreeNeuralNetwork(NeuralNetwork *nNet) {
    FreeLayer(nNet->inputLayer);
    FreeLayer(nNet->outputLayer);
    for (unsigned int i = 0; i<nNet->hiddenLayerCount; i++) {
        FreeLayer(nNet->hiddenLayers[i]);
    }
    free(nNet->name);
    free(nNet);
}

//precondition: Input layer is initialised
void AddHiddenLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > NET_LAYERS_ADDED) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = NET_LAYERS_ADDED;

    nNet->hiddenLayerCount++;
    Layer** temp = malloc(sizeof(Layer*) * nNet->hiddenLayerCount);
    memcpy(temp, nNet->hiddenLayers, sizeof(Layer*) * (nNet->hiddenLayerCount-1));
    free(nNet->hiddenLayers);
    nNet->hiddenLayers = temp;
    nNet->hiddenLayers[nNet->hiddenLayerCount-1] = layer;
}

void SetInputLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > NET_LAYERS_ADDED) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = NET_LAYERS_ADDED;
    nNet->inputLayer = layer;
}

//precondition: Hidden layers are finalised
void SetOutputLayer(NeuralNetwork *nNet, Layer *layer) {
    if (nNet->stage > NET_LAYERS_ADDED) {
        fprintf(stderr, "Error: Network layers have been finalized");
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = NET_LAYERS_ADDED;
    nNet->outputLayer = layer;
}

void FinalizeNeuralNetworkLayers(NeuralNetwork *nNet) {
    if (nNet->stage != NET_LAYERS_ADDED) {
        fprintf(stderr, "Error: Attempted to finalize layers of network in stage %d", nNet->stage);
        exit(EXIT_FAILURE_CODE);
    }
    nNet->stage = NET_LAYERS_FINALIZED;

    InitIOLayerTensor(nNet->inputLayer);

    // The first hidden layer's tensors are initialised outside the for loop as the previous layer will be the input layer;
    InitHiddenLayerTensor(nNet->inputLayer, nNet->hiddenLayers[0]);
    if (nNet->hiddenLayers[0]->uType == LINEAR_LAYER) {
        nNet->hiddenLayers[0]->linearLayer->weights = NewHeRandomMatrix(nNet->hiddenLayers[0]->computedValues->size, nNet->inputLayer->computedValues->size);
        nNet->hiddenLayers[0]->linearLayer->weightGradients = NewFilledMatrix(nNet->hiddenLayers[0]->computedValues->size, nNet->inputLayer->computedValues->size, 0);
    }

    for (int i = 1; i<nNet->hiddenLayerCount; i++) {
        InitHiddenLayerTensor(nNet->hiddenLayers[i-1], nNet->hiddenLayers[i]);
        if (nNet->hiddenLayers[i]->uType == LINEAR_LAYER) {
            nNet->hiddenLayers[i]->linearLayer->weights = NewHeRandomMatrix(nNet->hiddenLayers[i]->computedValues->size, nNet->hiddenLayers[i-1]->computedValues->size);
            nNet->hiddenLayers[i]->linearLayer->weightGradients = NewFilledMatrix(nNet->hiddenLayers[i]->computedValues->size, nNet->hiddenLayers[i-1]->computedValues->size, 0);
        }
    }

    InitIOLayerTensor(nNet->outputLayer);
} //Call after adding layers

Tensor* RunNeuralNetwork(NeuralNetwork *nNet, Tensor *input) {
    /* commented out because training also uses this for forward run.
    if (nNet->stage < 3) {
        //fprintf(stderr, "Warning: Attempted to run untrained neural network (stage %d). Required stage: >= 3", nNet->stage);
        //exit(EXIT_FAILURE_CODE);
    }
    */
    if (nNet->stage < NET_LAYERS_FINALIZED) {
        fprintf(stderr, "Error: Attempted to run un-finalized neural network (stage %d). Required stage: >= 2", nNet->stage);
        exit(EXIT_FAILURE_CODE);
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

Tensor* NetworkTrainingStep(NeuralNetwork *nNet, Tensor *input, int expected, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)) {
    //convolutional neural network debug
    /*
    if (nNet->trainingStepCount % 200 == 0) {
        int i = 0;
        while (nNet->hiddenLayers[i]->uType != CONVOLUTION_LAYER) {
            i++;
        }
        printf("bias:\n");
        PrintVectorHorizontal(nNet->hiddenLayers[i]->convolutionLayer->kernelBiases);
        printf("\n\nprinting kernels:\n");
        for (int j = 0; j<nNet->hiddenLayers[i]->convolutionLayer->kernelCount; j++) {
            PrintMatrix(nNet->hiddenLayers[i]->convolutionLayer->kernelGradients[j]);
        }
        printf("\n\nprinting kernels done:\n");
    }
    */

    nNet->trainingStepCount++;

    Tensor *netOutput = RunNeuralNetwork(nNet, input); //don't free netOutput. just a reference


    double loss = lossFunction(netOutput, nNet->outputLayer->computedValueGradients, expected);

    if (nNet->trainingStepCount % 20 == 0) {
        printf("\n===== Training Step %8d =====\n", nNet->trainingStepCount);
        printf("Actual: %d. Prediction: %d. Prediction Weight: %f\nOutput Vector: ", expected, GetTensorMaxIndex(netOutput), GetTensorValuePos(netOutput, GetTensorMaxIndex(netOutput)));
        PrintVectorHorizontal(netOutput->vector);
        printf("Loss: %f\n", loss);
        printf("==================================\n\n");
    }

    ComputeIOLayerGradients(nNet->hiddenLayers[nNet->hiddenLayerCount-1], nNet->outputLayer);

    for (int i = nNet->hiddenLayerCount - 1; i >= 0; i--) {
        Layer *currentLayer = nNet->hiddenLayers[i];
        Layer *previousLayer = (i == 0) ? nNet->inputLayer : nNet->hiddenLayers[i-1];
        Layer *nextLayer = (i == nNet->hiddenLayerCount - 1) ? nNet->outputLayer : nNet->hiddenLayers[i+1];

        ComputeHiddenLayerGradients(previousLayer, currentLayer, nextLayer, nNet->learningRate);
    }

    return netOutput;
}

void TrainNetwork(NeuralNetwork *nNet, CSVFile *csvTrain, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)) {
    if (nNet->stage < NET_LAYERS_FINALIZED) {
        fprintf(stderr, "Error: Attempted to train un-finalized neural network (stage %d). Required stage: >= 2", nNet->stage);
        exit(EXIT_FAILURE_CODE);
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
        }
        RewindCSV(csvTrain);
    }

    FreeMnistDigit(mnistDigit);

    nNet->stage = NET_TRAINED;
}

void TestNetwork(NeuralNetwork  *nNet, CSVFile *csvTest, double (*lossFunction)(Tensor *tOutput, Tensor *tOutputGradient, int expected)) {
    if (nNet->stage < NET_TRAINED) {
        fprintf(stderr, "Warning: Attempted to test un-trained neural network (stage %d). Required stage: >= 3", nNet->stage);
        //exit(EXIT_FAILURE_CODE);
    }

    MnistDigit *mnistDigit = NewMnistDigit();
    RewindCSV(csvTest);
    SkipLine(csvTest);
    unsigned int correct = 0;
    unsigned int errors = 0;

    for (int i = 0; i<csvTest->rows; i++) {
        ReadDigitFromCSV(csvTest, mnistDigit);

        Tensor* inputTensor = NewTensorCloneMatrix(mnistDigit->pixels); //needs to be fred
        Tensor *netOutput = RunNeuralNetwork(nNet, inputTensor); //don't free netOutput. just a reference

        double loss = lossFunction(netOutput, nNet->outputLayer->computedValueGradients, mnistDigit->digit);

        if (mnistDigit->digit == GetTensorMaxIndex(netOutput)) {
            correct++;
        } else {
            errors++;
        }

        FreeTensor(inputTensor); //maybe optimize this later?
    }

    printf("\n========== Test Results: ==========\n");
    printf("Tested %d digits. Correct: %d, Incorrect: %d\n", correct + errors, correct, errors);
    printf("Total Accuracy: %.3f%%\n", 100.0 * (double)correct / ((double)correct + (double)errors));
    printf("===================================\n\n");

    FreeMnistDigit(mnistDigit);
}

int NeuralNetworkMain() {

    NeuralNetwork *nNet = NewNeuralNetwork(MNIST_TRAINING_ROUNDS, MNIST_MAX_TRAINING_STEPS, MNIST_LEARNING_RATE, "MNIST Test");

    SetInputLayer(nNet, NewMatrixLayer(28, 28));

    // Convolution Network

    AddHiddenLayer(nNet, NewReshapeLayer());
    AddHiddenLayer(nNet, NewConvolutionLayer(16, 3));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewPoolingLayer(2, MAX_POOLING));
    AddHiddenLayer(nNet, NewFlatteningLayer());
    //AddHiddenLayer(nNet, NewLinearLayer(512));
    //AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(256));
    AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    //AddHiddenLayer(nNet, NewLinearLayer(128));
    //AddHiddenLayer(nNet, NewElementWiseLayer(Relu, ReluPrime));
    AddHiddenLayer(nNet, NewLinearLayer(10));
    AddHiddenLayer(nNet, NewSoftMaxLayer());


    // Non-Convolutional Network
    /*
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
    */

    SetOutputLayer(nNet, NewVectorLayer(10));

    FinalizeNeuralNetworkLayers(nNet);

    // CSV-MNIST
    CSVFile *csvTrain; //csv file with data used to train the network (size = 60000)
    CSVFile *csvTest; //csv file with data used to test the trained network (size = 10000)

    if (PLATFORM == PLATFORM_WINDOWS) {
        csvTrain = OpenCSVFile("..\\data\\mnist_train.csv");
        csvTest = OpenCSVFile("..\\data\\mnist_test.csv");
    } else {
        csvTrain = OpenCSVFile("../data/mnist_train.csv");
        csvTest = OpenCSVFile("../data/mnist_test.csv");
    }

    TrainNetwork(nNet, csvTrain, CalculateMnistLoss);
    PrintNeuralNetworkInfo(nNet);
    TestNetwork(nNet, csvTest, CalculateMnistLoss);



    //Cleanup
    FreeNeuralNetwork(nNet);
    FreeCSV(csvTrain);
    FreeCSV(csvTest);

    return 0;
}
