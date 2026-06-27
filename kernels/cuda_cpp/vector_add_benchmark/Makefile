NVCC ?= nvcc
CUDA_ARCH ?= native
CXXFLAGS ?= -O3 -std=c++23

TARGET := vector_add_bench
SRC := vector_add_bench.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(CUDA_ARCH) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
