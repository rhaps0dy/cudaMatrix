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

void matPrint(Matrix *m)
{
	unsigned int x, y;
	const unsigned int MAX_SANE_WIDTH = 8;
	char answer;

	printf("\n");
	if(m->w>MAX_SANE_WIDTH)
	{
		printf("Matrix is very big, %dx%d. Are you sure you want to print it? [y/n] ", m->w, m->w);
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

	for(y=0; y < m->w; y++)
	{
		for(x=0; x < m->w; x++)
			printf("%f\t", m->h[y*m->w+x]);
		printf("\n");
	}
	printf("\n");
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

int matDecomposeLU(Matrix *src, Matrix *l, Matrix *u)
{
	if(src->w != l->w || l->w != u->w)
	{
		SPIT("Matrices must be of the same size\n");
		return -1;
	}
	//temporary
	CHECK(matFill(l));
	CHECK(matFill(u));
	return 0;
}

