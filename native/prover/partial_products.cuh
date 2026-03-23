// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_PARTIAL_PRODUCTS_CUH__
#define __ZEKNOX_PROVER_PARTIAL_PRODUCTS_CUH__

#include <cstddef>
#include <cstdint>

#ifdef USE_CUDA
#include <cuda_runtime.h>

/** Max routed wires supported per row in the quotient kernel (stack allocation). */
static constexpr size_t PARTIAL_PRODUCTS_MAX_ROUTED_WIRES = 256u;

/** Number of quotient chunk products per row (ceil(num_routed / quotient_degree_factor)). */
static inline size_t partial_products_num_chunks(size_t num_routed_wires,
                                               size_t quotient_degree_factor)
{
    if (quotient_degree_factor == 0) {
        return 0;
    }
    return (num_routed_wires + quotient_degree_factor - 1) / quotient_degree_factor;
}

/**
 * Phase A: one thread per row. Computes chunk-wise products of wire permutation quotients.
 * chunk_products_out layout: row-major [degree x num_chunks].
 * Pointers and scalars are canonical Goldilocks u64 (same layout as gl64_t on device).
 */
__global__ void compute_quotient_chunk_products_kernel(
    const uint64_t *wire_values,
    const uint64_t *sigmas,
    const uint64_t *subgroup,
    const uint64_t *k_is,
    uint64_t beta,
    uint64_t gamma,
    size_t degree,
    size_t num_wires,
    size_t num_routed_wires,
    size_t quotient_degree_factor,
    size_t num_chunks,
    uint64_t *chunk_products_out);

/**
 * Phase B (CPU, recommended first): sequential prefix product for Z and cumulative partials.
 * Copies chunk products from device, computes on host, copies partial_products and z_poly back.
 *
 * partial_products_out: [degree x num_chunks]; row i column k is
 *   z_poly[i] * prod_{t=0..k} chunk_products[i][t].
 * z_poly_out[i] is Z at the start of row i; z_poly[0] = 1; z_poly[i+1] = last partial of row i.
 */
void compute_z_prefix_product_host(
    const uint64_t *d_chunk_products,
    size_t degree,
    size_t num_chunks,
    uint64_t *d_partial_products_out,
    uint64_t *d_z_poly_out,
    cudaStream_t stream = 0);

/**
 * Phase B (GPU): single-block kernel; matches `compute_z_prefix_product_host` without device↔host copies.
 */
void launch_compute_z_prefix_product_gpu(
    const uint64_t *d_chunk_products,
    size_t degree,
    size_t num_chunks,
    uint64_t *d_partial_products_out,
    uint64_t *d_z_poly_out,
    cudaStream_t stream = 0);

/**
 * Pack Z + partial columns (row-major partials) into polynomial-major layout for `PolynomialBatchGPU::from_values`.
 */
void launch_pack_zs_pp_polynomials(
    const uint64_t *d_z_start,
    const uint64_t *d_partial_row_major,
    uint64_t *d_out_poly_major,
    size_t degree,
    size_t num_chunks,
    cudaStream_t stream = 0);

void launch_compute_quotient_chunk_products(
    const uint64_t *d_wire_values,
    const uint64_t *d_sigmas,
    const uint64_t *d_subgroup,
    const uint64_t *d_k_is,
    uint64_t beta,
    uint64_t gamma,
    size_t degree,
    size_t num_wires,
    size_t num_routed_wires,
    size_t quotient_degree_factor,
    uint64_t *d_chunk_products_out,
    cudaStream_t stream = 0);

/** Host-only reference for tests / debugging (matches GPU + prefix logic). */
void partial_products_cpu_reference(
    const uint64_t *wire_values,
    const uint64_t *sigmas,
    const uint64_t *subgroup,
    const uint64_t *k_is,
    uint64_t beta,
    uint64_t gamma,
    size_t degree,
    size_t num_wires,
    size_t num_routed_wires,
    size_t quotient_degree_factor,
    uint64_t *chunk_out,
    uint64_t *z_out,
    uint64_t *partial_out);

#endif // USE_CUDA

#endif // __ZEKNOX_PROVER_PARTIAL_PRODUCTS_CUH__
