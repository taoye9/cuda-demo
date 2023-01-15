TARGET := gemm

all: $(TARGET)

$(TARGET): gemm.cu
	nvcc -arch sm_70  --ptxas-options=-v  gemm.cu -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
