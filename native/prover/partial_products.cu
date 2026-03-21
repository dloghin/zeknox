// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <cuda_runtime.h>

#include "partial_products.cuh"
#include "types/int_types.h"
#include "ff/goldilocks.hpp"
#include "prover/gl64_ext2.cuh"

#ifdef USE_CUDA

static constexpr int QUOTIENT_KERNEL_BLOCK = 256;

__device__ static void batch_inv_denoms(
    const gl64_t *den,
    gl64_t *inv_out,
    size_t n,
    gl64_t *prefix)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        inv_out[0] = gl64_t::one() / den[0];
        return;
    }
    prefix[0] = den[0];
    for (size_t j = 1; j < n; ++j) {
        prefix[j] = prefix[j - 1] * den[j];
    }
    gl64_t inv = gl64_t::one() / prefix[n - 1];
    for (size_t i = n - 1; i > 0; --i) {
        inv_out[i] = inv * prefix[i - 1];
        inv = inv * den[i];
    }
    inv_out[0] = inv;
}

__global__ void compute_quotient_chunk_products_kernel(
    const uint64_t *wire_values,
    const uint64_t *sigmas,
    const uint64_t *subgroup,
    const uint64_t *k_is,
    uint64_t beta_u64,
    uint64_t gamma_u64,
    size_t degree,
    size_t num_wires,
    size_t num_routed_wires,
    size_t quotient_degree_factor,
    size_t num_chunks,
    uint64_t *chunk_products_out)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (row >= degree) {
        return;
    }
    if (num_routed_wires > PARTIAL_PRODUCTS_MAX_ROUTED_WIRES || quotient_degree_factor == 0) {
        return;
    }

    gl64_t beta = gl64_t(beta_u64);
    gl64_t gamma = gl64_t(gamma_u64);

    gl64_t numerator[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    gl64_t denoms[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    gl64_t inv_den[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    gl64_t prefix[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];

    gl64_t x = gl64_t(subgroup[row]);

    for (size_t j = 0; j < num_routed_wires; ++j) {
        gl64_t w = gl64_t(wire_values[row * num_wires + j]);
        gl64_t kj = gl64_t(k_is[j]);
        numerator[j] = w + beta * (kj * x) + gamma;
        denoms[j] = w + beta * gl64_t(sigmas[row * num_routed_wires + j]) + gamma;
    }

    batch_inv_denoms(denoms, inv_den, num_routed_wires, prefix);

    gl64_t quotients[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    for (size_t j = 0; j < num_routed_wires; ++j) {
        quotients[j] = numerator[j] * inv_den[j];
    }

    for (size_t c = 0; c < num_chunks; ++c) {
        size_t start = c * quotient_degree_factor;
        if (start >= num_routed_wires) {
            chunk_products_out[row * num_chunks + c] = (uint64_t)gl64_t::one();
            continue;
        }
        size_t end = start + quotient_degree_factor;
        if (end > num_routed_wires) {
            end = num_routed_wires;
        }
        gl64_t prod = gl64_t::one();
        for (size_t j = start; j < end; ++j) {
            prod = prod * quotients[j];
        }
        chunk_products_out[row * num_chunks + c] = (uint64_t)prod;
    }
}

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
    cudaStream_t stream)
{
    if (degree == 0 || num_routed_wires == 0 || quotient_degree_factor == 0) {
        return;
    }
    if (num_routed_wires > PARTIAL_PRODUCTS_MAX_ROUTED_WIRES) {
        return;
    }
    size_t num_chunks = partial_products_num_chunks(num_routed_wires, quotient_degree_factor);
    if (num_chunks == 0) {
        return;
    }

    int threads = QUOTIENT_KERNEL_BLOCK;
    int blocks = (int)((degree + (size_t)threads - 1) / (size_t)threads);
    compute_quotient_chunk_products_kernel<<<blocks, threads, 0, stream>>>(
        d_wire_values,
        d_sigmas,
        d_subgroup,
        d_k_is,
        beta,
        gamma,
        degree,
        num_wires,
        num_routed_wires,
        quotient_degree_factor,
        num_chunks,
        d_chunk_products_out);
}

#ifndef __CUDA_ARCH__

void compute_z_prefix_product_host(
    const uint64_t *d_chunk_products,
    size_t degree,
    size_t num_chunks,
    uint64_t *d_partial_products_out,
    uint64_t *d_z_poly_out,
    cudaStream_t stream)
{
    if (degree == 0 || num_chunks == 0) {
        return;
    }

    const size_t total_chunk = degree * num_chunks;
    std::vector<uint64_t> h_chunk(total_chunk);
    std::vector<uint64_t> h_partial(total_chunk);
    std::vector<uint64_t> h_z(degree);

    cudaMemcpy(
        h_chunk.data(),
        d_chunk_products,
        total_chunk * sizeof(uint64_t),
        cudaMemcpyDeviceToHost);

    using F = cpp_gl64_t;
    F z_row = F::one();
    h_z[0] = z_row.get_val();

    for (size_t i = 0; i < degree; ++i) {
        F cum = F::one();
        for (size_t k = 0; k < num_chunks; ++k) {
            F ck = F(h_chunk[i * num_chunks + k]);
            cum = cum * ck;
            F partial = z_row * cum;
            h_partial[i * num_chunks + k] = partial.get_val();
        }
        F z_next = F(h_partial[i * num_chunks + (num_chunks - 1)]);
        if (i + 1 < degree) {
            z_row = z_next;
            h_z[i + 1] = z_row.get_val();
        }
    }

    cudaMemcpyAsync(
        d_partial_products_out,
        h_partial.data(),
        total_chunk * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream);
    cudaMemcpyAsync(
        d_z_poly_out,
        h_z.data(),
        degree * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        stream);
    cudaStreamSynchronize(stream);
}

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
    uint64_t *partial_out)
{
    using F = cpp_gl64_t;
    size_t num_chunks = partial_products_num_chunks(num_routed_wires, quotient_degree_factor);
    F beta_f = F(beta);
    F gamma_f = F(gamma);

    for (size_t row = 0; row < degree; ++row) {
        std::vector<F> quotients(num_routed_wires);
        F x = F(subgroup[row]);
        for (size_t j = 0; j < num_routed_wires; ++j) {
            F w = F(wire_values[row * num_wires + j]);
            F num = w + beta_f * (F(k_is[j]) * x) + gamma_f;
            F den = w + beta_f * F(sigmas[row * num_routed_wires + j]) + gamma_f;
            quotients[j] = num * inv_base(den);
        }
        for (size_t c = 0; c < num_chunks; ++c) {
            size_t start = c * quotient_degree_factor;
            size_t end = start + quotient_degree_factor;
            if (end > num_routed_wires) {
                end = num_routed_wires;
            }
            if (start >= num_routed_wires) {
                chunk_out[row * num_chunks + c] = F::one().get_val();
                continue;
            }
            F prod = F::one();
            for (size_t j = start; j < end; ++j) {
                prod = prod * quotients[j];
            }
            chunk_out[row * num_chunks + c] = prod.get_val();
        }
    }

    F z_row = F::one();
    z_out[0] = z_row.get_val();
    for (size_t i = 0; i < degree; ++i) {
        F cum = F::one();
        for (size_t k = 0; k < num_chunks; ++k) {
            F ck = F(chunk_out[i * num_chunks + k]);
            cum = cum * ck;
            F partial = z_row * cum;
            partial_out[i * num_chunks + k] = partial.get_val();
        }
        F z_next = F(partial_out[i * num_chunks + (num_chunks - 1)]);
        if (i + 1 < degree) {
            z_row = z_next;
            z_out[i + 1] = z_row.get_val();
        }
    }
}

#endif // !__CUDA_ARCH__

#endif // USE_CUDA
