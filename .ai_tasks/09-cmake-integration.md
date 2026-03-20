# Task 09: CMake Build Integration

## Objective

Add the `native/prover/` source files to the `native/CMakeLists.txt` build system so the prover is compiled as part of `libzeknox.a`.

## Background

The existing build system (`native/CMakeLists.txt`) compiles:
- `zeknox_cpp` target: C++ objects (no CUDA)
- `zeknox_cuda` target: CUDA objects (when `USE_CUDA=ON`)
- `zeknox` target: static library combining both

Source files are organized by module: `ntt/`, `merkle/`, `poseidon/`, etc.

## Deliverables

### Modifications to `native/CMakeLists.txt`

Add the prover source files to the appropriate targets:

```cmake
# CUDA sources for the prover
if(USE_CUDA)
    list(APPEND CUDA_SOURCES
        prover/partial_products.cu
        prover/quotient_poly.cu
        prover/fri_fold.cu
        prover/opening_set.cu
        prover/prover.cu
    )
endif()

# C++ sources for the prover (host-only code)
list(APPEND CPP_SOURCES
    # challenger.hpp is header-only, no .cpp needed unless separated
)
```

### Include Path

Ensure `native/prover/` headers are accessible:
```cmake
target_include_directories(zeknox_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Already includes the root native/ directory, so #include <prover/prover.h> works
```

### Update `native/lib.h`

Add the prover C API to the main header:
```cpp
#include <prover/prover.h>
```

### Dependencies

The prover module depends on:
- `ntt/` (NTT, LDE, transpose)
- `merkle/` (Merkle tree)
- `poseidon/` (Poseidon hash for challenger)
- `ff/` (Goldilocks field)
- `primitives/` (extension field)

These are all already compiled into `libzeknox.a`, so no additional linking is needed.

## Testing

- Verify `cmake .. -DUSE_CUDA=ON -DCURVE=gl64` builds without errors
- Verify `libzeknox.a` includes the prover symbols (check with `nm`)
- Run existing tests to ensure no regressions
