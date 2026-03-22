// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_CUDA_GRID_LIMITS_CUH__
#define __ZEKNOX_CUDA_GRID_LIMITS_CUH__

#include <climits>
#include <cstddef>

#include "exception.hpp"

// CUDA kernel launches use int / dim3 grid dimensions; casting a larger block count
// from size_t to int truncates silently. Compute blocks without overflow in (extent + tpb - 1) / tpb
// and require the result fits in int.
inline int zeknox_cuda_grid_blocks_int(size_t extent, unsigned block_size, const char *context)
{
    if (extent == 0 || block_size == 0) {
        return 0;
    }
    size_t num_blocks = extent / block_size + (extent % block_size != 0 ? 1u : 0u);
    if (num_blocks > (size_t)INT_MAX) {
        throw zeknox_error{-1, fmt("%s: grid size %zu blocks exceeds INT_MAX (%d)",
            context ? context : "kernel", num_blocks, INT_MAX)};
    }
    return static_cast<int>(num_blocks);
}

#endif // __ZEKNOX_CUDA_GRID_LIMITS_CUH__
