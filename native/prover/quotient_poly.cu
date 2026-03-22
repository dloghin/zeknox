// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <cuda_runtime.h>

#include "quotient_poly.cuh"

#ifdef USE_CUDA

#include "utils/cuda_grid_limits.cuh"
#include "types/int_types.h"
#include "ff/goldilocks.hpp"
#include "prover/gl64_ext2.cuh"

static constexpr int QUOTIENT_KERNEL_BLOCK = 256;

__device__ static gl64_t gl64_pow_u64(gl64_t base, uint64_t exp)
{
    gl64_t r = gl64_t::one();
    gl64_t b = base;
    while (exp > 0) {
        if (exp & 1ULL) {
            r = r * b;
        }
        b = b * b;
        exp >>= 1;
    }
    return r;
}

__global__ void precompute_z_h_inverse_kernel(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t degree_bits,
    uint64_t *z_h_inv_out,
    size_t lde_size)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= lde_size) {
        return;
    }

    gl64_t omega = gl64_t(omega_lde_u64);
    gl64_t g = gl64_t(coset_shift_u64);
    gl64_t wi = gl64_pow_u64(omega, (uint64_t)i);
    gl64_t x = g * wi;
    uint64_t n = 1ULL << degree_bits;
    gl64_t xn = gl64_pow_u64(x, n);
    gl64_t z_h = xn - gl64_t::one();
    if ((uint64_t)z_h == 0ULL) {
        z_h_inv_out[i] = 0;
        return;
    }
    gl64_t inv = gl64_t::one() / z_h;
    z_h_inv_out[i] = (uint64_t)inv;
}

void launch_precompute_z_h_inverse(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t degree_bits,
    uint64_t *d_z_h_inv,
    size_t lde_size,
    cudaStream_t stream)
{
    if (lde_size == 0) {
        return;
    }
    int blocks = zeknox_cuda_grid_blocks_int(lde_size, (unsigned)QUOTIENT_KERNEL_BLOCK, "launch_precompute_z_h_inverse");
    precompute_z_h_inverse_kernel<<<blocks, QUOTIENT_KERNEL_BLOCK, 0, stream>>>(
        coset_shift_u64,
        omega_lde_u64,
        degree_bits,
        d_z_h_inv,
        lde_size);
}

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
    size_t num_challenges)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= lde_size) {
        return;
    }
    if (num_challenges < 1) {
        return;
    }

    size_t nalpha = num_alpha_weights;
    if (nalpha > num_gate_constraints) {
        nalpha = num_gate_constraints;
    }

    gl64_t acc = gl64_t::zero();
    for (size_t j = 0; j < nalpha; ++j) {
        gl64_t g = gl64_t(gate_constraint_values[j * lde_size + i]);
        gl64_t w = gl64_t(alpha_weights[j]);
        acc = acc + g * w;
    }
    if (extra_present) {
        acc = acc + gl64_t(extra_vanishing_terms[i]);
    }
    gl64_t zh = gl64_t(z_h_inv[i]);
    gl64_t q = acc * zh;
    quotient_values_out[0 * lde_size + i] = (uint64_t)q;
    for (size_t c = 1; c < num_challenges; ++c) {
        quotient_values_out[c * lde_size + i] = 0;
    }
}

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
    cudaStream_t stream)
{
    if (lde_size == 0 || num_challenges == 0) {
        return;
    }
    int blocks = zeknox_cuda_grid_blocks_int(lde_size, (unsigned)QUOTIENT_KERNEL_BLOCK, "launch_eval_vanishing_poly");
    eval_vanishing_poly_kernel<<<blocks, QUOTIENT_KERNEL_BLOCK, 0, stream>>>(
        d_gate_constraint_values,
        num_gate_constraints,
        lde_size,
        d_alpha_weights,
        num_alpha_weights,
        d_extra_vanishing_terms,
        extra_present,
        d_z_h_inv,
        d_quotient_values_out,
        num_challenges);
}

#ifndef __CUDA_ARCH__

static cpp_gl64_t cpp_pow_u64(cpp_gl64_t base, uint64_t exp)
{
    cpp_gl64_t r = cpp_gl64_t::one();
    cpp_gl64_t b = base;
    while (exp > 0) {
        if (exp & 1ULL) {
            r = r * b;
        }
        b = b * b;
        exp >>= 1;
    }
    return r;
}

void quotient_precompute_z_h_inverse_cpu(
    uint64_t coset_shift_u64,
    uint64_t omega_lde_u64,
    uint32_t lde_log,
    uint32_t degree_bits,
    uint64_t *z_h_inv_out,
    size_t lde_size)
{
    (void)lde_log;
    cpp_gl64_t omega = cpp_gl64_t(omega_lde_u64);
    cpp_gl64_t g = cpp_gl64_t(coset_shift_u64);
    uint64_t n = 1ULL << degree_bits;
    for (size_t i = 0; i < lde_size; ++i) {
        cpp_gl64_t wi = cpp_pow_u64(omega, (uint64_t)i);
        cpp_gl64_t x = g * wi;
        cpp_gl64_t xn = cpp_pow_u64(x, n);
        cpp_gl64_t z_h = xn - cpp_gl64_t::one();
        if (z_h.get_val() == 0) {
            z_h_inv_out[i] = 0;
            continue;
        }
        z_h_inv_out[i] = inv_base(z_h).get_val();
    }
}

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
    size_t num_challenges)
{
    size_t nalpha = num_alpha_weights;
    if (nalpha > num_gate_constraints) {
        nalpha = num_gate_constraints;
    }
    for (size_t i = 0; i < lde_size; ++i) {
        cpp_gl64_t acc = cpp_gl64_t::zero();
        for (size_t j = 0; j < nalpha; ++j) {
            cpp_gl64_t g = cpp_gl64_t(gate_constraint_values[j * lde_size + i]);
            cpp_gl64_t w = cpp_gl64_t(alpha_weights[j]);
            acc = acc + g * w;
        }
        if (extra_present) {
            acc = acc + cpp_gl64_t(extra_vanishing_terms[i]);
        }
        cpp_gl64_t zh = cpp_gl64_t(z_h_inv[i]);
        cpp_gl64_t q = acc * zh;
        quotient_out[0 * lde_size + i] = q.get_val();
        for (size_t c = 1; c < num_challenges; ++c) {
            quotient_out[c * lde_size + i] = 0;
        }
    }
}

#endif // !__CUDA_ARCH__

#endif // USE_CUDA
