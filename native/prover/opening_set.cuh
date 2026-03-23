// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_OPENING_SET_CUH__
#define __ZEKNOX_PROVER_OPENING_SET_CUH__

#include <cstddef>
#include <vector>

#include <prover/gl64_ext2.cuh>
#include <prover/polynomial_batch.cuh>

#ifdef __cplusplus

/// Plonky2 `OpeningSet`-style container: extension-field evaluations of committed polynomials
/// at `zeta` and (for Z) at `g * zeta`.
struct OpeningSet {
    std::vector<gl64_ext2_t> constants;
    std::vector<gl64_ext2_t> plonk_sigmas;
    std::vector<gl64_ext2_t> wires;
    std::vector<gl64_ext2_t> plonk_zs;
    std::vector<gl64_ext2_t> plonk_zs_next;
    std::vector<gl64_ext2_t> partial_products;
    std::vector<gl64_ext2_t> quotient_polys;
};

/// Evaluate all committed polynomials needed for the opening set (GPU Horner at `zeta` / `g*zeta`).
/// Index ranges are half-open `[start, end)` into the corresponding `PolynomialBatchGPU::coeffs_gpu`
/// layout (same `degree = 2^degree_log` coefficients per polynomial, ascending order).
OpeningSet construct_opening_set(
    gl64_ext2_t zeta,
    gl64_ext2_t g,
    const PolynomialBatchGPU &constants_sigmas,
    const PolynomialBatchGPU &wires,
    const PolynomialBatchGPU &zs_partial_products,
    const PolynomialBatchGPU &quotient_polys,
    size_t constants_start,
    size_t constants_end,
    size_t sigmas_start,
    size_t sigmas_end,
    size_t zs_start,
    size_t zs_end,
    size_t partial_products_start,
    size_t partial_products_end,
    size_t gpu_id);

#endif // __cplusplus

#endif // __ZEKNOX_PROVER_OPENING_SET_CUH__
