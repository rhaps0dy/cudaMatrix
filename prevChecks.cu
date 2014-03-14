#include "prevChecks.hu"

#include "errorMacros.hu"

int checkCLIArguments(int argc, char *argv[], unsigned int *width)
{
	unsigned int i, ovfCheck, curDigit;

	//basic syntax and positive integer
	if(argc!=2)
	{
		SPIT("Usage: %s <matrix_width>\n", argv[0]);
		return -1;
	}

	(*width) = 0;
	for(i=0; i<strlen(argv[1]); i++)
	{
		(*width) *= 10;
		if(argv[1][i]<'0' || argv[1][i]>'9')
		{
			SPIT("Could not parse \"%s\" as positive integer\n", argv[1]);
			return -1;
		}
		//sum with overflow check
		curDigit = (unsigned int) (argv[1][i]-'0');
		ovfCheck = (*width);
		(*width) += curDigit;
		if((*width) < curDigit | (*width) < ovfCheck)
		{
			SPIT("Integer \"%s\" too big, overflows\n", argv[1]);
			return -1;
		}
	}
	return 0;
}

int checkCUDAPresent(void)
{
	int count;
	cudaGetDeviceCount(&count);
	if(count>0)
		return 0;
	SPIT("No CUDA capable devices found!\n");
	return -1;
}
