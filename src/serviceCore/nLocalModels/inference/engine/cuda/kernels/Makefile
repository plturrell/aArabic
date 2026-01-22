# Makefile for CUDA Kernels (Dequantization + Transformer)
# Compiles to shared libraries for Zig FFI integration

NVCC := nvcc
CUDA_ARCH := -arch=sm_75  # Tesla T4 (Turing)

# Compiler flags
NVCC_FLAGS := -shared -Xcompiler '-fPIC' $(CUDA_ARCH) -O3

# Source and output files
DEQUANT_SRC := dequant_kernels.cu
DEQUANT_LIB := libdequant_kernels.so

TRANSFORMER_SRC := transformer_kernels.cu
TRANSFORMER_LIB := libtransformer_kernels.so

# Install directories
PREFIX ?= /usr/local
LIBDIR := $(PREFIX)/lib

.PHONY: all clean install test debug compat help

all: $(DEQUANT_LIB) $(TRANSFORMER_LIB)

$(DEQUANT_LIB): $(DEQUANT_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✅ Built $(DEQUANT_LIB)"

$(TRANSFORMER_LIB): $(TRANSFORMER_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "✅ Built $(TRANSFORMER_LIB)"

clean:
	rm -f $(DEQUANT_LIB) $(TRANSFORMER_LIB)
	@echo "✅ Cleaned"

install: all
	mkdir -p $(LIBDIR)
	cp $(DEQUANT_LIB) $(TRANSFORMER_LIB) $(LIBDIR)/
	ldconfig 2>/dev/null || true
	@echo "✅ Installed to $(LIBDIR)/"

# Test build with verbose output
test: all
	@echo "Testing dequant kernel library..."
	@nm -D $(DEQUANT_LIB) | grep dequant || echo "Warning: No dequant symbols found"
	@echo ""
	@echo "Testing transformer kernel library..."
	@nm -D $(TRANSFORMER_LIB) | grep cuda_ || echo "Warning: No cuda_ symbols found"
	@echo ""
	@echo "Exported symbols (dequant):"
	@nm -D $(DEQUANT_LIB) | grep " T " | head -10
	@echo ""
	@echo "Exported symbols (transformer):"
	@nm -D $(TRANSFORMER_LIB) | grep " T " | head -10

# Debug build
debug: NVCC_FLAGS += -g -G
debug: all

# For systems without sm_75 (fallback)
compat: CUDA_ARCH := -arch=sm_70
compat: all

help:
	@echo "CUDA Kernels Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build all kernel libraries (default)"
	@echo "  clean   - Remove built files"
	@echo "  install - Install to $(LIBDIR)"
	@echo "  test    - Build and show exported symbols"
	@echo "  debug   - Build with debug symbols"
	@echo "  compat  - Build for sm_70 compatibility"
	@echo ""
	@echo "Libraries:"
	@echo "  $(DEQUANT_LIB)     - Dequantization kernels"
	@echo "  $(TRANSFORMER_LIB) - Transformer kernels (RMSNorm, SiLU, etc.)"
	@echo ""
	@echo "Environment:"
	@echo "  PREFIX  - Installation prefix (default: /usr/local)"

