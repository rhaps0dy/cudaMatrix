#ifndef _cuMatrix_hu_
#define _cuMatrix_hu_

__global__ void _matMultiply(float *dest, float *a, float *b, unsigned int width);
__global__ void _matDifferent(float *a, float *b, unsigned int width, const float tolerance, bool *result);

__device__ float _getLValue(float *m, unsigned int y, unsigned int x, unsigned int width);
__device__ float _getUValue(float *m, unsigned int y, unsigned int x, unsigned int width);
__global__ void _matMultiplyLU(float *dest, float *lu, unsigned int width);
__global__ void _matMultiplyUL(float *dest, float *lu, unsigned int width);

__global__ void _makeLURow(float *m, unsigned int y, unsigned int initPos, unsigned int width);
__global__ void _prepareLeftColLU(float *m, unsigned int width);

__global__ void _doInversionStepUpper(float *dest, float *m, unsigned int width, unsigned int i);
__global__ void _doInversionStepLower(float *dest, float *m, unsigned int width, unsigned int i);
#endif //_cuMatrix_hu_
