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

	if(hasBeenTouched())
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
	// doolittle algorithm
	unsigned int x, y, p;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(1, w-1);
	_prepareLeftColLU<<<dimGrid, dimBlock>>>(d, w);
	copyDtoH();

	dimBlock = dim3(w-1, 1);
	_makeLURow<<<dimGrid, dimBlock>>>(d, 1, w+1, w);
	//copy recent changes to host
	CHECK_SUCCESS(cudaMemcpy(&h[w+1], &d[w+1], (w-1)*sizeof(float), cudaMemcpyDeviceToHost)); 

	for(y=2; y<w; y++)
	{
		for(x=1; x<y; x++)
		{
			for(p=0; p<x; p++)
				h[y*w+x] -= h[y*w+p]*h[p*w+x];
			h[y*w+x] /= h[x*w+x];
		}
		//copy recent changes to device
		CHECK_SUCCESS(cudaMemcpy(&d[y*w+1], &h[y*w+1], (y-1)*sizeof(float), cudaMemcpyHostToDevice)); 

		dimBlock = dim3(w-y, 1);
		_makeLURow<<<dimGrid, dimBlock>>>(d, y, y*w+y, w);
		//copy recent changes to host
		CHECK_SUCCESS(cudaMemcpy(&h[y*w+y], &d[y*w+y], (w-y)*sizeof(float), cudaMemcpyDeviceToHost)); 
	}

	setLU(true);
}

void Matrix::multiply(Matrix *a, Matrix *b)
{
	if(w != a->getW() || w != b->getW())
		SPIT("Matrices must be of the same size\n");

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	_matMultiply<<<dimGrid, dimBlock>>>(d, a->getD(), b->getD(), w);

	setLU(false);

	touch();
}

void Matrix::multiplyLU()
{
	float *dcopy;
	size_t size = w*w*sizeof(float);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	CHECK_SUCCESS(cudaMalloc((void **)&dcopy, size));
	CHECK_SUCCESS(cudaMemcpy(dcopy, d, size, cudaMemcpyDeviceToDevice));

	_matMultiplyLU<<<dimGrid, dimBlock>>>(d, dcopy, w);
	cudaFree(dcopy);
	setLU(false);
	
	touch();
}

void Matrix::multiplyUL()
{
	float *dcopy;
	size_t size = w*w*sizeof(float);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(w, w);

	CHECK_SUCCESS(cudaMalloc((void **)&dcopy, size));
	CHECK_SUCCESS(cudaMemcpy(dcopy, d, size, cudaMemcpyDeviceToDevice));

	_matMultiplyUL<<<dimGrid, dimBlock>>>(d, dcopy, w);
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

void Matrix::invertLU()
{
	unsigned int i;
	/*float *dest, *m = (float *) malloc(w*w*sizeof(float));
	memcpy((void *)m, (const void *)h, w*w*sizeof(float));
	dest = h;
	CHECK_SUCCESS(cudaMalloc((void **)&m, w*w*sizeof(float)));
	for(i=0; i<w; i++)
	{
		min = (w-i < i+1 ? w-i : i+1);
		dim3 dimGrid(1, 1);
		dim3 dimBlock(min, min);
		_doInversionStepUpper<<<dimGrid, dimBlock>>>(m, d, w, i);
		_doInversionStepLower<<<dimGrid, dimBlock>>>(m, d, w, i);
	}
	CHECK_SUCCESS(cudaMemcpy(d, m, w*w*sizeof(float), cudaMemcpyDeviceToDevice));
	copyDtoH();
	cudaFree(m);*/

	//hardcoded 3x3 matrix inversion
	if(w != 3)
		SPIT("You're supposed to use a 3x3 matrix only! Support for bigger matrices soon in https://github.com/rhaps0dy\n");
	copyDtoH();
	//upper

	h[2] = (((h[1]*h[5])/(h[4]*h[8]))-h[2]/h[8])/h[0];
	for(i=0; i<3; i++)
		h[i*w+i] = 1/h[i*w+i];

	for(i=0; i<2; i++)
		h[i*w+i+1] = -h[i*w+i+1]*h[i*w+i]*h[(i+1)*w+i+1];


	//lower
	for(i=0; i<2; i++)
		h[(i+1)*w+i] = -h[(i+1)*w+i];
	h[2*w] = h[1*w]*h[2*w+1] - h[2*w];
	copyHtoD();

	
//	free(m);
}
