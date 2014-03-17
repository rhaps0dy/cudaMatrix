#include "cuMatrix.hu"

//! Populate the scope with current position
#define ESTABLISH_CURRENT_POSITION unsigned int y, x, pos; \
	y = blockIdx.y*blockDim.y + threadIdx.y; \
	x = blockIdx.x*blockDim.x + threadIdx.x; \
	pos = y*width+x;

__global__ void _doInversionStepUpper(float *dest, float *m, unsigned int width, unsigned int i)
{
	//no need to calculate pos
	unsigned int y, x;
	y = -blockIdx.y*blockDim.y - threadIdx.y;
	x = blockIdx.x*blockDim.x + threadIdx.x;

	if(y==0)
	{
		if(x==0)
		{
			dest[i*width+i] = 1/m[i*width+i];
			return;
		}
		dest[i*width+i+x] = m[i*width+i+x]*m[i*width+i];
		return;
	}

	if(x==0)
	{
		dest[(i+y)*width+i] = -(m[(i+y)*width+i]*m[i*width+i]);
		return;
	}
	dest[(i+y)*width+x+i] = m[(i+y)*width+x+i] + m[(i+y)*width+i] * m[i*width+i+x];
}

__global__ void _doInversionStepLower(float *dest, float *m, unsigned int width, unsigned int i)
{
	unsigned int y, x;
	y = blockIdx.y*blockDim.y + threadIdx.y;
	x = -blockIdx.x*blockDim.x - threadIdx.x;

	if(y==0)
	{
		if(x==0)
			return;
		
		dest[i*width+i+x] = m[i*width+i+x];
		return;
	}

	if(x==0)
	{
		dest[(i+y)*width+i] = -m[(i+y)*width+i];
		return;
	}
	dest[(i+y)*width+x+i] = m[(i+y)*width+x+i] + m[(i+y)*width+i] * m[i*width+i+x];
}

__global__ void _prepareLeftColLU(float *m, unsigned int width)
{
	unsigned int pos = width*(blockIdx.y*blockDim.y + threadIdx.y+1); 
	m[pos] /= m[0];
}

__global__ void _makeLURow(float *m, unsigned int y, unsigned int initPos, unsigned int width)
{
	unsigned int i, x = blockIdx.x*blockDim.x + threadIdx.x; 
	for(i=0; i<y; i++)
		m[initPos+x] -= m[y*width+i]*m[i*width+x+y];
}

__global__ void _matMultiply(float *dest, float *a, float *b, unsigned int width)
{
	unsigned int i;
	ESTABLISH_CURRENT_POSITION;

	dest[pos] = a[y*width] * b[x];
	for(i=1; i<width; i++)
		dest[pos] += a[y*width+i]*b[i*width+x];
}
__global__ void _matMultiplyLU(float *dest, float *lu, unsigned int width)
{
	unsigned int i;
	ESTABLISH_CURRENT_POSITION;

	dest[pos] = _getLValue(lu, y, 0, width) * _getUValue(lu, 0, x, width);
	for(i=1; i<width; i++)
		dest[pos] += _getLValue(lu, y, i, width) * _getUValue(lu, i, x, width);
}

__global__ void _matMultiplyUL(float *dest, float *lu, unsigned int width)
{
	unsigned int i;
	ESTABLISH_CURRENT_POSITION;

	dest[pos] = _getUValue(lu, y, 0, width) * _getLValue(lu, 0, x, width);
	for(i=1; i<width; i++)
		dest[pos] += _getUValue(lu, y, i, width) * _getLValue(lu, i, x, width);
}

__device__ float _getLValue(float *m, unsigned int y, unsigned int x, unsigned int width)
{
	if(y<x) return 0.0;
	if(y==x) return 1.0;
	return m[y*width+x];
}
__device__ float _getUValue(float *m, unsigned int y, unsigned int x, unsigned int width)
{
	if(y>x) return 0.0;
	return m[y*width+x];
}

__global__ void _matDifferent(float *a, float *b, unsigned int width, const float tolerance, bool *result)
{
	ESTABLISH_CURRENT_POSITION;
	float diff = a[pos]-b[pos];

	if(diff < -tolerance || diff > tolerance)
		*result = true;
}
