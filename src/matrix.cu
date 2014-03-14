#include "matrix.hu"

#include <stdio.h>
#include "errorMacros.hu"
#include "cuMatrix.hu"

int matFill(Matrix *m)
{ 
	unsigned int i, num;
	int divide;
	FILE *f;

	f=fopen("/dev/urandom", "r");
	ASSERT(f);
	for(i=0; i< m->w*m->w; i++)
	{
		fread(&num, sizeof(unsigned int), 1, f);
		fread(&divide, sizeof(int), 1, f);
		m->h[i] = ((float)num)/((float)divide);
	}
	CHECK(fclose(f));

	// sync matrix and host memories
	CHECK(_matCopyHtoD(m));
	return 0;
} 

int matPrint(Matrix *m)
{
	unsigned int x, y;
	const unsigned int MAX_SANE_WIDTH = 6;
	char answer;

	printf("\n");
	if(m->w>MAX_SANE_WIDTH)
	{
		printf("Matrix is very big, %dx%d. Are you sure you want to print it? [y/n] ", m->w, m->w);
		while(1)
		{
			scanf("%c", &answer);
			if(answer=='n')
				return 0;
			if(answer=='y')
				break;
			printf("Could not understand input. Please type 'y' or 'n'. ");
		} 
	}

	CHECK(_matCopyDtoH(m))

	for(y=0; y < m->w; y++)
	{
		for(x=0; x < m->w; x++)
			printf("%f ", m->h[y*m->w+x]);
		printf("\n");
	}
	printf("\n");
	return 0;
}

Matrix *matAlloc(unsigned int width)
{
	Matrix *m;
	size_t size = width*width*sizeof(float);

	m = (Matrix *) malloc(sizeof(Matrix));
	if(!m)
	{
		SPIT("Failed to allocate Matrix\n");
		return NULL;
	}

	m->w = width;
	m->touched = 0;

	if(cudaMalloc((void **)&m->d, size)!=cudaSuccess)
	{
		SPIT("Failed to allocate device array\n");
		free(m);
		return NULL;
	}

	m->h = (float *) malloc(size);
	if(!m->h)
	{
		SPIT("Failed to allocate host array\n");
		free(m);
		cudaFree(m->d);
		return NULL;
	}

	return m;
}

void matFree(Matrix *m)
{
	free(m->h);
	cudaFree(m->d);
	free(m);
}

void __matDecomposeLU(float* src, float *l, float *u, unsigned int width)
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

	for(i=0; i<width; i++)
		for(j=0; j<width; j++)
		{
			if(j<i)
			{
				l[i*width+j] = src[i*width+j];
				u[i*width+j] = 0;
				continue;
			}
			if(i==j)
				l[i*width+j] = 1;
			else l[i*width+j] = 0;
				u[i*width+j] = src[i*width+j];
		}
}

int matDecomposeLU(Matrix *src, Matrix *l, Matrix *u)
{
	if(src->w != l->w || l->w != u->w)
	{
		SPIT("Matrices must be of the same size\n");
		return -1;
	}

	//dim3 dimGrid(1, 1);
	//dim3 dimBlock(src->w, src->w);

	//_matDecomposeLU<<<dimGrid, dimBlock>>>(src->d, l->d, u->d, src->w);
	__matDecomposeLU(src->h, l->h, u->h, src->w);

	_matCopyHtoD(l);
	_matCopyHtoD(u);

	l->touched = 0;
	u->touched = 0;
	return 0;
}

int matMultiply(Matrix *a, Matrix *b, Matrix *dest)
{
	if(dest->w != a->w || a->w != b->w)
	{
		SPIT("Matrices must be of the same size\n");
		return -1;
	}

	dim3 dimGrid(1, 1);
	dim3 dimBlock(dest->w, dest->w);

	_matMultiply<<<dimGrid, dimBlock>>>(a->d, b->d, dest->d, dest->w);

	dest->touched = 1;
	return 0;
}
