// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_QUOTIENT_POLY_CUH__
#define __ZEKNOX_PROVER_QUOTIENT_POLY_CUH__

#include <cstddef>
#include <cstdint>

#ifdef USE_CUDA
#include <cuda_runtime.h>

/**
 * For trace subgroup order n = 2^degree_bits, vanishing polynomial evaluated at x is Z_H(x) = x^n - 1.
 * Coset LDE points: x_i = coset_shift * omega^i where omega is a primitive lde_size root (lde_size = 2^lde_log).
 * One thread per LDE index i (linear ordering; matches tests / simple CPU reference).
 */
__global__ void precompute_z_h_inverse_kernel(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t degree_bits,
    uint64_t *z_h_inv_out,
    size_t lde_size);

void launch_precompute_z_h_inverse(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t degree_bits,
    uint64_t *d_z_h_inv,
    size_t lde_size,
    cudaStream_t stream = 0);

/**
 * Combine gate constraint rows with alpha_weights, add optional per-point extra terms, then multiply by z_h_inv.
 * Layout gate_constraint_values[row * lde_size + point] for row in [0, num_gate_constraints).
 * quotient_out[challenge * lde_size + point] — for now num_challenges must be 1 (placeholder for multi-challenge wiring).
 */
__global__ void eval_vanishing_poly_kernel(
    const uint64_t *gate_constraint_values,
    size_t num_gate_constraints,
    size_t lde_size,
    const uint64_t *alpha_weights,
    size_t num_alpha_weights,
    const uint64_t *extra_vanishing_terms,
    int extra_present,
    const uint64_t *z_h_inv,
    uint64_t *quotient_values_out,
    size_t num_challenges);

void launch_eval_vanishing_poly(
    const uint64_t *d_gate_constraint_values,
    size_t num_gate_constraints,
    size_t lde_size,
    const uint64_t *d_alpha_weights,
    size_t num_alpha_weights,
    const uint64_t *d_extra_vanishing_terms,
    int extra_present,
    const uint64_t *d_z_h_inv,
    uint64_t *d_quotient_values_out,
    size_t num_challenges,
    cudaStream_t stream = 0);

/** Host reference for tests (linear coset indexing). */
void quotient_precompute_z_h_inverse_cpu(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t lde_log,
    uint32_t degree_bits,
    uint64_t *z_h_inv_out,
    size_t lde_size);

void quotient_eval_vanishing_cpu(
    const uint64_t *gate_constraint_values,
    size_t num_gate_constraints,
    size_t lde_size,
    const uint64_t *alpha_weights,
    size_t num_alpha_weights,
    const uint64_t *extra_vanishing_terms,
    int extra_present,
    const uint64_t *z_h_inv,
    uint64_t *quotient_out,
    size_t num_challenges);

#endif // USE_CUDA

#endif // __ZEKNOX_PROVER_QUOTIENT_POLY_CUH__
