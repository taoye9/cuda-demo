# Object file directory:
OBJ_DIR = bin

OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/gemm_cpu.o $(OBJ_DIR)/cublas_gemm.o
INCLUDES = gemm.h

#c++ compiler options
CXX = nvcc
CXX_FLAGS = -g -std=c++14

CUDA_LINK_LIBS= -lcudart -lcublas

#NVCC compiler options
NVCC=nvcc
NVCC_FLAGS= -g -std=c++14 -arch=sm_70 -gencode=arch=compute_70,code=sm_70 --ptxas-options=-v

TARGET := test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $@ $(CUDA_LINK_LIBS)

# compile c++ source code
$(OBJ_DIR)/%.o : %.cpp $(INCLUDES)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : %.cu $(INCLUDES)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
