#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "errorMacros.hu"
#include "prevChecks.hu"
#include "matrix.hu"

int main(int argc, char *argv[])
{
	Matrix *m, *l, *u;
	unsigned int width;

	//some preliminary checks
	CHECK(checkCLIArguments(argc, argv, &width));
	CHECK(checkCUDAPresent());

	printf("Matrix width and height: %u\n", width);

	printf("Allocating memory...\n");
	m = matAlloc(width);
	ASSERT(m);
	l = matAlloc(width);
	ASSERT(l);
	u = matAlloc(width);
	ASSERT(u);

	printf("Filling matrix with random numbers...\n");
	CHECK(matFill(m));
	
	printf("Matrix\n");
	CHECK(matPrint(m));

	matDecomposeLU(m, l, u);

	printf("Lower\n");
	CHECK(matPrint(l));
	printf("Upper\n");
	CHECK(matPrint(u));

	//matMultiply(l, u, m);
//	printf("Identity:\n");
//	matPrint(m);

	printf("Freeing memory...\n");
	matFree(m);
	matFree(l);
	matFree(u);
	return 0;
}
