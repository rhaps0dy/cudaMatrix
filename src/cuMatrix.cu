#include "cuMatrix.hu"

//! Populate the scope with current position
#define ESTABLISH_CURRENT_POSITION unsigned int y, x, pos; \
	y = blockIdx.y*blockDim.y + threadIdx.y; \
	x = blockIdx.x*blockDim.x + threadIdx.x; \
	pos = y*width+x;

void __matDecomposeLU(float* src, unsigned int width)
{
	unsigned int i,j,p;
	for(i=0; i<width; i++)
	{
		for(j=0; j<i; j++)
		{
			float a = src[i*width+j];
			for(p=0; p<j; p++)
				a = a-src[i*width+p]*src[p*width+j];
			src[i*width+j] = a/src[j*width+j];
		}
		for(j=i; j<width; j++)
		{
			float a = src[i*width+j];
			for(p=0; p<i; p++)
				a = a-src[i*width+p]*src[p*width+j];
			src[i*width+j] = a;
		}
	}
}/*

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
}*/

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
