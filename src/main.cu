#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "errorMacros.hu"
#include "prevChecks.hu"
#include "matrix.hu"

int main(int argc, char *argv[])
{
	Matrix *m, *lu, *inv;
	unsigned int width;

	//some preliminary checks
	checkCLIArguments(argc, argv, &width);
	checkCUDAPresent();

	printf("Matrix width and height: %u\n", width);

	printf("Allocating memory...\n");
	m = new Matrix(width);

	printf("Filling matrix with random numbers...\n");
	m->fill();
	
	printf("Matrix\n");
	m->print();

	lu = m->copy();
	lu->decomposeLU();

	printf("LU Matrix:\n");
	lu->print();

	inv = lu->copy();

	lu->composeLU();
	printf("Original matrix check:\n");
	lu->print();

	if(m->isDifferent(lu))
		printf("Matrices DIFFER\n");
	else
		printf("Matrices don't differ, all is correct!\n");

	inv->invertLU();
	printf("Inverted LU\n");
	inv->print();
	inv->composeLU();
	printf("Original matrix check:\n");
	lu->multiply(m, inv);
	lu->print();

	printf("Freeing memory...\n");
	delete lu;
	delete m;
	delete inv;
	return 0;
}
