// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cstring>

#include <utils/gpu_t.cuh>
#include <utils/all_gpus.hpp>
#include <ntt/ntt.cuh>
#include <merkle/merkle.h>
#include <ff/goldilocks.hpp>

#include "polynomial_batch.cuh"

static inline __host__ __device__ size_t reverse_bits(size_t val, size_t bit_count) {
    size_t result = 0;
    for (size_t i = 0; i < bit_count; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// ---------- PolynomialBatchGPU lifetime ----------

PolynomialBatchGPU::PolynomialBatchGPU()
    : lde_gpu(nullptr), num_leaves(0), leaf_size(0),
      digests_gpu(nullptr), cap_gpu(nullptr), num_digests(0), cap_len(0),
      coeffs_gpu(nullptr), num_polynomials(0),
      degree_log(0), rate_bits(0), blinding(false), cap_height(0),
      gpu_id(0), owns_coeffs(false) {}

void PolynomialBatchGPU::release() noexcept {
    if (lde_gpu)     { cudaFree(lde_gpu); }
    if (digests_gpu) { cudaFree(digests_gpu); }
    if (cap_gpu)     { cudaFree(cap_gpu); }
    if (owns_coeffs && coeffs_gpu) { cudaFree(coeffs_gpu); }
}

PolynomialBatchGPU::~PolynomialBatchGPU() {
    release();
}

PolynomialBatchGPU::PolynomialBatchGPU(PolynomialBatchGPU &&other) noexcept
    : lde_gpu(other.lde_gpu), num_leaves(other.num_leaves), leaf_size(other.leaf_size),
      digests_gpu(other.digests_gpu), cap_gpu(other.cap_gpu),
      num_digests(other.num_digests), cap_len(other.cap_len),
      coeffs_gpu(other.coeffs_gpu), num_polynomials(other.num_polynomials),
      degree_log(other.degree_log), rate_bits(other.rate_bits),
      blinding(other.blinding), cap_height(other.cap_height),
      gpu_id(other.gpu_id), owns_coeffs(other.owns_coeffs) {
    other.lde_gpu = nullptr;
    other.digests_gpu = nullptr;
    other.cap_gpu = nullptr;
    other.coeffs_gpu = nullptr;
    other.owns_coeffs = false;
}

PolynomialBatchGPU &PolynomialBatchGPU::operator=(PolynomialBatchGPU &&other) noexcept {
    if (this != &other) {
        release();

        lde_gpu = other.lde_gpu;
        num_leaves = other.num_leaves;
        leaf_size = other.leaf_size;
        digests_gpu = other.digests_gpu;
        cap_gpu = other.cap_gpu;
        num_digests = other.num_digests;
        cap_len = other.cap_len;
        coeffs_gpu = other.coeffs_gpu;
        num_polynomials = other.num_polynomials;
        degree_log = other.degree_log;
        rate_bits = other.rate_bits;
        blinding = other.blinding;
        cap_height = other.cap_height;
        gpu_id = other.gpu_id;
        owns_coeffs = other.owns_coeffs;

        other.lde_gpu = nullptr;
        other.digests_gpu = nullptr;
        other.cap_gpu = nullptr;
        other.coeffs_gpu = nullptr;
        other.owns_coeffs = false;
    }
    return *this;
}

// ---------- Internal: LDE + transpose + Merkle ----------

void PolynomialBatchGPU::build_lde_and_merkle(
    size_t num_polys, size_t deg_log, size_t r_bits,
    bool blind, size_t c_height, size_t g_id)
{
    this->num_polynomials = num_polys;
    this->degree_log = deg_log;
    this->rate_bits = r_bits;
    this->blinding = blind;
    this->cap_height = c_height;
    this->gpu_id = g_id;

    size_t salt = blind ? SALT_SIZE : 0;
    size_t output_domain_log = deg_log + r_bits;
    size_t output_domain_size = (size_t)1 << output_domain_log;
    size_t num_cols = num_polys + salt;
    size_t total_output_elems = num_cols * output_domain_size;

    auto &gpu = select_gpu(g_id);
    gpu.select();

    // 1. LDE: coefficients -> coset evaluations
    NTT_Config lde_cfg = {};
    lde_cfg.batches = (uint32_t)num_polys;
    lde_cfg.order = NR;
    lde_cfg.ntt_type = standard;
    lde_cfg.extension_rate_bits = (uint32_t)r_bits;
    lde_cfg.are_inputs_on_device = true;
    lde_cfg.are_outputs_on_device = true;
    lde_cfg.with_coset = true;
    lde_cfg.is_multi_gpu = false;
    lde_cfg.salt_size = (uint32_t)salt;

    fr_t *lde_output = nullptr;
    CUDA_OK(cudaMalloc(&lde_output, total_output_elems * sizeof(fr_t)));

    RustError err = ntt::batch_lde(gpu, lde_output, this->coeffs_gpu,
                                   (uint32_t)deg_log, forward, lde_cfg);
    if (err.code != 0) {
        cudaFree(lde_output);
        throw cuda_error{err.code, err.message ? err.message : "batch_lde failed"};
    }

    // 2. Transpose + bit-reverse
    NTT_TransposeConfig trans_cfg = {};
    trans_cfg.batches = (uint32_t)num_cols;
    trans_cfg.are_inputs_on_device = true;
    trans_cfg.are_outputs_on_device = true;

    fr_t *transposed = nullptr;
    CUDA_OK(cudaMalloc(&transposed, total_output_elems * sizeof(fr_t)));

    err = ntt::compute_transpose_rev(gpu, transposed, lde_output,
                                     (uint32_t)output_domain_log, trans_cfg);
    cudaFree(lde_output);
    if (err.code != 0) {
        cudaFree(transposed);
        throw cuda_error{err.code, err.message ? err.message : "transpose_rev failed"};
    }

    this->lde_gpu = transposed;
    this->num_leaves = output_domain_size;
    this->leaf_size = num_cols;

    // 3. Merkle tree
    this->cap_len = (size_t)1 << c_height;
    this->num_digests = 2 * (output_domain_size - this->cap_len);

    size_t cap_alloc = this->cap_len * NUM_HASH_OUT_ELTS;

    if (this->num_digests == 0) {
        this->digests_gpu = nullptr;
    } else {
        CUDA_OK(cudaMalloc(&this->digests_gpu, this->num_digests * NUM_HASH_OUT_ELTS * sizeof(fr_t)));
    }
    CUDA_OK(cudaMalloc(&this->cap_gpu, cap_alloc * sizeof(fr_t)));

    fill_digests_buf_linear_gpu_with_gpu_ptr(
        (void *)this->digests_gpu,
        (void *)this->cap_gpu,
        (void *)this->lde_gpu,
        (u64)this->num_digests,
        (u64)this->cap_len,
        (u64)output_domain_size,
        (u64)num_cols,
        (u64)c_height,
        (u64)HashPoseidon,
        (u64)g_id
    );

    gpu.sync();
}

// ---------- Public static factories ----------

PolynomialBatchGPU PolynomialBatchGPU::from_values(
    fr_t *values_gpu, size_t num_polys,
    size_t deg_log, size_t r_bits,
    bool blind, size_t c_height, size_t g_id)
{
    auto &gpu = select_gpu(g_id);
    gpu.select();

    // IFFT: values -> coefficients (in-place)
    NTT_Config ntt_cfg = {};
    ntt_cfg.batches = (uint32_t)num_polys;
    ntt_cfg.order = NN;
    ntt_cfg.ntt_type = standard;
    ntt_cfg.extension_rate_bits = 0;
    ntt_cfg.are_inputs_on_device = true;
    ntt_cfg.are_outputs_on_device = true;
    ntt_cfg.with_coset = false;
    ntt_cfg.is_multi_gpu = false;
    ntt_cfg.salt_size = 0;

    RustError err = ntt::batch_ntt(gpu, values_gpu, (uint32_t)deg_log, inverse, ntt_cfg);
    if (err.code != 0) {
        throw cuda_error{err.code, err.message ? err.message : "batch_ntt (IFFT) failed"};
    }

    // Now values_gpu holds coefficients; build from_coeffs
    PolynomialBatchGPU batch;
    batch.coeffs_gpu = values_gpu;
    batch.owns_coeffs = false;
    batch.build_lde_and_merkle(num_polys, deg_log, r_bits, blind, c_height, g_id);
    return batch;
}

PolynomialBatchGPU PolynomialBatchGPU::from_coeffs(
    fr_t *coeffs_gpu_in, size_t num_polys,
    size_t deg_log, size_t r_bits,
    bool blind, size_t c_height, size_t g_id)
{
    PolynomialBatchGPU batch;
    batch.coeffs_gpu = coeffs_gpu_in;
    batch.owns_coeffs = false;
    batch.build_lde_and_merkle(num_polys, deg_log, r_bits, blind, c_height, g_id);
    return batch;
}

// ---------- LDE value access ----------

size_t PolynomialBatchGPU::lde_values_offset(size_t index, size_t step) const {
    size_t total_bits = this->degree_log + this->rate_bits;
    size_t raw = index * step;
    size_t rev = reverse_bits(raw, total_bits);
    return rev * this->leaf_size;
}

// ---------- Host copy helpers ----------

void PolynomialBatchGPU::copy_cap_to_host(fr_t *host_cap, size_t max_elems) const {
    size_t n = this->cap_len * NUM_HASH_OUT_ELTS;
    if (n > max_elems) n = max_elems;
    if (n == 0 || !this->cap_gpu) return;
    CUDA_OK(cudaMemcpy(host_cap, this->cap_gpu, n * sizeof(fr_t),
                        cudaMemcpyDeviceToHost));
}

void PolynomialBatchGPU::copy_digests_to_host(fr_t *host_digests, size_t max_elems) const {
    size_t n = this->num_digests * NUM_HASH_OUT_ELTS;
    if (n > max_elems) n = max_elems;
    if (n == 0 || !this->digests_gpu) return;
    CUDA_OK(cudaMemcpy(host_digests, this->digests_gpu, n * sizeof(fr_t),
                        cudaMemcpyDeviceToHost));
}

// ---------- C API ----------

#ifndef __CUDA_ARCH__

extern "C" RustError polynomial_batch_from_values(
    size_t device_id,
    void *values_gpu,
    uint32_t num_polys,
    uint32_t degree_log,
    uint32_t rate_bits,
    int blinding,
    uint32_t cap_height,
    void **out_lde_gpu,
    void **out_digests_gpu,
    void **out_cap_gpu,
    void **out_coeffs_gpu,
    uint64_t *out_num_leaves,
    uint64_t *out_leaf_size,
    uint64_t *out_num_digests,
    uint64_t *out_cap_len)
{
    try {
        auto batch = PolynomialBatchGPU::from_values(
            (fr_t *)values_gpu, num_polys, degree_log,
            rate_bits, blinding != 0, cap_height, device_id);

        *out_lde_gpu = (void *)batch.lde_gpu;
        *out_digests_gpu = (void *)batch.digests_gpu;
        *out_cap_gpu = (void *)batch.cap_gpu;
        *out_coeffs_gpu = (void *)batch.coeffs_gpu;
        *out_num_leaves = batch.num_leaves;
        *out_leaf_size = batch.leaf_size;
        *out_num_digests = batch.num_digests;
        *out_cap_len = batch.cap_len;

        // Transfer ownership: prevent destructor from freeing GPU memory
        batch.lde_gpu = nullptr;
        batch.digests_gpu = nullptr;
        batch.cap_gpu = nullptr;
        batch.coeffs_gpu = nullptr;

        return RustError{0};
    } catch (const std::exception &e) {
        return RustError{-1, e.what()};
    }
}

extern "C" RustError polynomial_batch_from_coeffs(
    size_t device_id,
    const void *coeffs_gpu,
    uint32_t num_polys,
    uint32_t degree_log,
    uint32_t rate_bits,
    int blinding,
    uint32_t cap_height,
    void **out_lde_gpu,
    void **out_digests_gpu,
    void **out_cap_gpu,
    uint64_t *out_num_leaves,
    uint64_t *out_leaf_size,
    uint64_t *out_num_digests,
    uint64_t *out_cap_len)
{
    try {
        auto batch = PolynomialBatchGPU::from_coeffs(
            (fr_t *)coeffs_gpu, num_polys, degree_log,
            rate_bits, blinding != 0, cap_height, device_id);

        *out_lde_gpu = (void *)batch.lde_gpu;
        *out_digests_gpu = (void *)batch.digests_gpu;
        *out_cap_gpu = (void *)batch.cap_gpu;
        *out_num_leaves = batch.num_leaves;
        *out_leaf_size = batch.leaf_size;
        *out_num_digests = batch.num_digests;
        *out_cap_len = batch.cap_len;

        batch.lde_gpu = nullptr;
        batch.digests_gpu = nullptr;
        batch.cap_gpu = nullptr;

        return RustError{0};
    } catch (const std::exception &e) {
        return RustError{-1, e.what()};
    }
}

#endif // __CUDA_ARCH__
