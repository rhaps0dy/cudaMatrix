#include "matrix.hu"

#include <stdio.h>
#include <stdlib.h>
#include "errorMacros.hu"
#include "cuMatrix.hu"

void Matrix::fill()
{ 
	unsigned int i, num;
	int divide;
	FILE *f;

	f=fopen("/dev/urandom", "r");
	ASSERT(f);
	for(i=0; i< w*w; i++)
	{
		fread(&num, sizeof(unsigned int), 1, f);
		fread(&divide, sizeof(int), 1, f);
		h[i] = ((float)num)/((float)divide);
	}
	CHECK(fclose(f));

	// sync matrix and host memories
	copyHtoD();
} 

void Matrix::print()
{
	unsigned int x, y;
	char answer;

	printf("\n");
	if(w>MAX_SANE_WIDTH)
	{
		printf("Matrix is very big, %dx%d. Are you sure you want to print it? [y/n] ", w, w);
		while(1)
		{
			scanf("%c", &answer);
			if(answer=='n')
				return;
			if(answer=='y')
				break;
			printf("Could not understand input. Please type 'y' or 'n'. ");
		} 
	}

	copyDtoH();

	if(!isLU())
	{
		for(y=0; y < w; y++)
		{
			for(x=0; x < w; x++)
				printf("%f ", h[y*w+x]);
			printf("\n");
		}
		printf("\n");
		return;
	}
	//print the L and the U matrices
	printf("Lower part\n");
	for(y=0; y < w; y++)
	{
		for(x=0; x < w; x++)
		{
			if(x<y)
				printf("%f ", h[y*w+x]);
			else if(x==y)
				printf("%f ", 1.0);
			else
				printf("%f ", 0.0);
		}
		printf("\n");
	}
	printf("\nUpper part\n");
	for(y=0; y < w; y++)
	{
		for(x=0; x < w; x++)
		{
			if(x>=y)
				printf("%f ", h[y*w+x]);
			else
				printf("%f ", 0.0);
		}
		printf("\n");
	}
	printf("\n");
}

Matrix::Matrix(unsigned int width) :
w(width), touched(false), _isLU(false)
{
	size_t size = width*width*sizeof(float);

	if(cudaMalloc((void **)&d, size)!=cudaSuccess)
		SPIT("Failed to allocate device array\n");

	h = (float *) malloc(size);
	if(!h)
	{
		cudaFree(d);
		SPIT("Failed to allocate host array\n");
	}
}

Matrix *Matrix::copy()
{
	size_t size = w*w*sizeof(float);
	Matrix *m = new Matrix(w);
	memcpy((void *)m->getH(), (const void *)h, size);
	CHECK_SUCCESS(cudaMemcpy(m->getD(), d, size, cudaMemcpyDeviceToDevice));
	m->setLU(isLU());
	return m;
}

Matrix::~Matrix()
{
	free(h);
	cudaFree(d);
}

void Matrix::decomposeLU()
{
	//dim3 dimGrid(1, 1);
	//dim3 dimBlock(src->w, src->w);

	//_matDecomposeLU<<<dimGrid, dimBlock>>>(src->d, l->d, u->d, src->w);
	__matDecomposeLU(h, w);
	setLU(true);
	copyHtoD();
}

void Matrix::multiply(Matrix *a, Matrix *b)
{
	if(w != a->getW() || w != b->getW())
		SPIT("Matrices must be of the same size\n");

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	_matMultiply<<<dimGrid, dimBlock>>>(d, a->getD(), b->getD(), w);

	touch();
}

void Matrix::composeLU()
{
	float *dcopy;
	size_t size = w*w*sizeof(float);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	CHECK_SUCCESS(cudaMalloc((void **)&dcopy, size));
	CHECK_SUCCESS(cudaMemcpy(dcopy, d, size, cudaMemcpyDeviceToDevice));

	_matComposeLU<<<dimGrid, dimBlock>>>(d, dcopy, w);
	cudaFree(dcopy);
	setLU(false);
	
	touch();
}

bool Matrix::isDifferent(Matrix *m)
{
	bool result, *dev_result;

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	CHECK_SUCCESS(cudaMalloc((void **)&dev_result, sizeof(bool)));
	result = false;
	CHECK_SUCCESS(cudaMemcpy(dev_result, &result, sizeof(bool), cudaMemcpyHostToDevice));
	_matDifferent<<<dimGrid, dimBlock>>>(d, m->getD(), w, 0.001, dev_result);
	CHECK_SUCCESS(cudaMemcpy(&result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost));
	cudaFree(dev_result);
	return result;
}
