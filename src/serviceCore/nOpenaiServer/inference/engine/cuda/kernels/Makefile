# Makefile for CUDA Dequantization Kernels
# Compiles to a shared library for Zig FFI integration

NVCC := nvcc
CUDA_ARCH := -arch=sm_75  # Tesla T4 (Turing)

# Compiler flags
NVCC_FLAGS := -shared -Xcompiler '-fPIC' $(CUDA_ARCH) -O3

# Source and output
SRC := dequant_kernels.cu
LIB := libdequant_kernels.so

# Install directories
PREFIX ?= /usr/local
LIBDIR := $(PREFIX)/lib

.PHONY: all clean install test

all: $(LIB)

$(LIB): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✅ Built $(LIB)"

clean:
	rm -f $(LIB)
	@echo "✅ Cleaned"

install: $(LIB)
	mkdir -p $(LIBDIR)
	cp $(LIB) $(LIBDIR)/
	ldconfig 2>/dev/null || true
	@echo "✅ Installed to $(LIBDIR)/$(LIB)"

# Test build with verbose output
test: $(LIB)
	@echo "Testing dequant kernel library..."
	@nm -D $(LIB) | grep dequant || echo "Warning: No dequant symbols found"
	@echo "Exported symbols:"
	@nm -D $(LIB) | grep " T " | head -10

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(LIB)

# For systems without sm_75 (fallback)
compat: CUDA_ARCH := -arch=sm_70
compat: $(LIB)

help:
	@echo "CUDA Dequantization Kernels Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build libdequant_kernels.so (default)"
	@echo "  clean   - Remove built files"
	@echo "  install - Install to $(LIBDIR)"
	@echo "  test    - Build and show exported symbols"
	@echo "  debug   - Build with debug symbols"
	@echo "  compat  - Build for sm_70 compatibility"
	@echo ""
	@echo "Environment:"
	@echo "  PREFIX  - Installation prefix (default: /usr/local)"

