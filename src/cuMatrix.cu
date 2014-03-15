#include "cuMatrix.hu"

//! Populate the scope with current position
#define ESTABLISH_CURRENT_POSITION unsigned int y, x, pos; \
	y = blockIdx.y*blockDim.y + threadIdx.y; \
	x = blockIdx.x*blockDim.x + threadIdx.x; \
	pos = y*width+x;

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
__global__ void _matComposeLU(float *dest, float *lu, unsigned int width)
{
	unsigned int i, maxi;
	ESTABLISH_CURRENT_POSITION;

	if(y>x)
	{
		maxi = x;
		dest[pos] = lu[y*width+x]*lu[x*width+x];
	}
	else
	{
		maxi = y;
		dest[pos] = lu[pos];
	}

	for(i=0; i<maxi; i++)
		dest[pos] += lu[y*width+i]*lu[i*width+x];
}

__global__ void _matDifferent(float *a, float *b, unsigned int width, const float tolerance, bool *result)
{
	ESTABLISH_CURRENT_POSITION;
	float diff = a[pos]-b[pos];

	if(diff < -tolerance || diff > tolerance)
		*result = true;
}
