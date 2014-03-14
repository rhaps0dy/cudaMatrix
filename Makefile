default: all
.PHONY: default
	
all:
	nvcc -o inverse_nxn main.cu prevChecks.cu matrix.cu
.PHONY: all

clean:
	rm inverse_nxn
.PHONY: all
