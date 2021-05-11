#ifndef _MULTI_PRODUCT_H
#define _MULTI_PRODUCT_H

#include <vector>
#include <cuda.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <assert.h>
#include <cufft.h>
#include "shorder.hpp"
#include "shproduct.h"
#include "select_size.hpp"
#include "sh.hpp"


const int traditional_blocksize = 256;
constexpr int N5 = select_size_5(n);
constexpr int N2 = select_size_2(n);
const std::vector<int> gammalist = {22,33,44,55,66,77};

void gpu_initGamma();
void releaseGamma();

void multi_product(float *A, float *B, float* C, float *D, float *E, float *F,
                    int multi_product_num, int type);
void shprod_many(float* A, float* B, float* C, float* D, float* E, float* F,
            cufftComplex* pool0, cufftComplex* pool1, cufftComplex* pool2,
            int multi_product_num, cufftHandle plan);
void shprod_many(float *A, float *B, float*C,
            cufftComplex* pool0, cufftComplex* pool1, cufftComplex* pool2,
            int num, cufftHandle plan);

#endif