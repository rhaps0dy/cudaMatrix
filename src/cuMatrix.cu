#include "cuMatrix.hu"

//! Populate the scope with current position
#define ESTABLISH_CURRENT_POSITION unsigned int y, x, pos; \
	y = blockIdx.y*blockDim.y + threadIdx.y; \
	x = blockIdx.x*blockDim.x + threadIdx.x; \
	pos = y*width+x;

__global__ void _matDecomposeLU(float* src, float *l, float *u, unsigned int width)
{
	unsigned int i;
	ESTABLISH_CURRENT_POSITION;

	if(x<y) //we are in the L part
	{
		u[pos] = 0.;
		l[pos] = src[pos];
		for(i=0; i<x; i++)
			l[pos] -= src[y*width+i]*src[i*width+x];
		l[pos] /= src[x*width+x];
		return;
	}

	//we are in the U part
	if(x==y)
		l[pos] = 1.;
	else l[pos] = 0.;

	u[pos] = src[pos];
	for(i=0; i<x; i++)
		u[pos] -= src[y*width+i]*src[i*width+x];
}

__global__ void _matMultiply(float *a, float *b, float *dest, unsigned int width)
{
	unsigned int i;
	ESTABLISH_CURRENT_POSITION;

	dest[pos] = a[y*width] * b[x];
	for(i=1; i<width; i++)
		dest[pos] += a[y*width+i]*b[i*width+x];
}
