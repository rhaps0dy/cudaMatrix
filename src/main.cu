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

	printf("Decomposing in LU...\n");
	lu = m->copy();
	lu->decomposeLU();
	lu->print();
	inv = lu->copy();
	printf("Inverting the LU...\n");
	inv->invertLU();
	inv->print();
	inv->multiplyUL();
	printf("Inverse = Inv(U) * Inv(L):\n");
	inv->print();
	printf("Multiply inverse by original matrix to check if it's identity:\n");
	lu->multiply(m, inv);
	lu->print();

	printf("Freeing memory...\n");
	delete lu;
	delete m;
	delete inv;
	return 0;
}
