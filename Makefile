default: all
.PHONY: default
	
all:
	nvcc -o inverse_nxn src/main.cu src/prevChecks.cu src/matrix.cu src/cuMatrix.cu
.PHONY: all

clean:
	rm inverse_nxn
.PHONY: all
