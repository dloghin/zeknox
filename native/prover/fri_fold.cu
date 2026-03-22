// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cmath>
#include <cstring>

#include <string>

#include <ff/goldilocks.hpp>
#include <merkle/merkle.h>
#include <merkle/hasher.hpp>
#include <ntt/ntt.cuh>
#include <ntt/ntt.h>
#include <prover/challenger.hpp>
#include <prover/gl64_ext2.cuh>
#include <utils/cuda_utils.cuh>
#include <utils/gpu_t.cuh>
#include <utils/all_gpus.hpp>

#include "fri_fold.cuh"

#ifndef __CUDA_ARCH__

/// Check `cudaMalloc` result and non-null pointer for non-zero size; throws `cuda_error` on failure.
static void cuda_malloc_bytes(void **ptr, size_t bytes)
{
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        if (ptr) {
            *ptr = nullptr;
        }
        throw cuda_error{-err, std::string("cudaMalloc(") + std::to_string(bytes) + " bytes) failed: " + cudaGetErrorString(err)};
    }
    if (bytes != 0 && (*ptr == nullptr)) {
        throw cuda_error{-cudaErrorUnknown, "cudaMalloc returned nullptr for non-zero size"};
    }
}

/// Allocate `count` `fr_t` elements on the device. When `count > 0`, checks allocation success and
/// returns a non-null pointer or throws `cuda_error`. When `count == 0`, returns nullptr.
static fr_t *cuda_malloc_fr_count(size_t count)
{
    if (count == 0) {
        return nullptr;
    }
    void *p = nullptr;
    cuda_malloc_bytes(&p, count * sizeof(fr_t));
    return static_cast<fr_t *>(p);
}

static void free_merkle_tree(MerkleTreeGPU &t)
{
    if (t.leaves_gpu) {
        cudaFree(t.leaves_gpu);
    }
    if (t.digests_gpu) {
        cudaFree(t.digests_gpu);
    }
    if (t.cap_gpu) {
        cudaFree(t.cap_gpu);
    }
    t.leaves_gpu = nullptr;
    t.digests_gpu = nullptr;
    t.cap_gpu = nullptr;
}

static void clear_merkle_trees(std::vector<MerkleTreeGPU> &trees)
{
    for (auto &t : trees) {
        free_merkle_tree(t);
    }
    trees.clear();
}

static inline size_t host_lg2(size_t n)
{
    size_t l = 0;
    while (((size_t)1 << l) < n) {
        ++l;
    }
    return l;
}

static cpp_gl64_t gl64_pow_u64(cpp_gl64_t base, uint64_t exp)
{
    cpp_gl64_t r = cpp_gl64_t::one();
    while (exp > 0) {
        if (exp & 1ULL) {
            r = r * base;
        }
        base = base * base;
        exp >>= 1;
    }
    return r;
}

#endif // !__CUDA_ARCH__

#if defined(__CUDACC__)

static __device__ __forceinline__ size_t d_reverse_bits(size_t val, size_t bit_count)
{
    size_t result = 0;
    for (size_t i = 0; i < bit_count; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

static __device__ __forceinline__ gl64_ext2_t d_load_ext2(const fr_t *base, size_t idx, size_t ext_degree)
{
    size_t o = idx * ext_degree;
    if (ext_degree == 2) {
        return gl64_ext2_t(base[o], base[o + 1]);
    }
    return gl64_ext2_t(base[o]);
}

static __device__ __forceinline__ void d_store_ext2(fr_t *base, size_t idx, size_t ext_degree, const gl64_ext2_t &x)
{
    size_t o = idx * ext_degree;
    if (ext_degree == 2) {
        base[o] = x.real;
        base[o + 1] = x.imag;
    } else {
        base[o] = x.real;
    }
}

__global__ void fri_prepare_merkle_leaves(
    const fr_t *values,
    fr_t *leaves_out,
    size_t n,
    size_t lg_n,
    size_t arity_bits,
    size_t ext_degree)
{
    size_t arity = (size_t)1 << arity_bits;
    size_t num_chunks = n / arity;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chunks) {
        return;
    }
    for (size_t j = 0; j < arity; j++) {
        size_t src_idx = d_reverse_bits(tid * arity + j, lg_n);
        for (size_t k = 0; k < ext_degree; k++) {
            leaves_out[tid * arity * ext_degree + j * ext_degree + k] =
                values[src_idx * ext_degree + k];
        }
    }
}

__global__ void fri_fold_coefficients(
    const fr_t *coeffs_in,
    fr_t *coeffs_out,
    const fr_t beta[2],
    size_t n,
    size_t arity_bits,
    size_t ext_degree)
{
    size_t arity = (size_t)1 << arity_bits;
    size_t num_out = n / arity;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_out) {
        return;
    }
    gl64_ext2_t b(beta[0], beta[1]);
    gl64_ext2_t acc = d_load_ext2(coeffs_in, tid * arity + arity - 1, ext_degree);
    for (int r = (int)arity - 2; r >= 0; r--) {
        acc = acc * b + d_load_ext2(coeffs_in, tid * arity + (size_t)r, ext_degree);
    }
    d_store_ext2(coeffs_out, tid, ext_degree, acc);
}

__global__ void fri_ext2_deinterleave(const fr_t *in, fr_t *out, size_t n)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    out[tid] = in[tid * 2];
    out[n + tid] = in[tid * 2 + 1];
}

__global__ void fri_ext2_interleave(const fr_t *in, fr_t *out, size_t n)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    out[tid * 2] = in[tid];
    out[tid * 2 + 1] = in[n + tid];
}

#endif // __CUDACC__

#ifndef __CUDA_ARCH__

static void launch_fri_prepare_merkle_leaves(
    const fr_t *values,
    fr_t *leaves_out,
    size_t n,
    size_t lg_n,
    size_t arity_bits,
    size_t ext_degree,
    cudaStream_t stream)
{
    size_t arity = (size_t)1 << arity_bits;
    size_t num_chunks = n / arity;
    int threads = 256;
    int blocks = (int)((num_chunks + (size_t)threads - 1) / (size_t)threads);
    fri_prepare_merkle_leaves<<<blocks, threads, 0, stream>>>(
        values, leaves_out, n, lg_n, arity_bits, ext_degree);
}

static void launch_fri_fold_coefficients(
    const fr_t *coeffs_in,
    fr_t *coeffs_out,
    const fr_t *beta,
    size_t n,
    size_t arity_bits,
    size_t ext_degree,
    cudaStream_t stream)
{
    size_t arity = (size_t)1 << arity_bits;
    size_t num_out = n / arity;
    int threads = 256;
    int blocks = (int)((num_out + (size_t)threads - 1) / (size_t)threads);
    fri_fold_coefficients<<<blocks, threads, 0, stream>>>(
        coeffs_in, coeffs_out, beta, n, arity_bits, ext_degree);
}

static RustError fri_ext2_coset_fft_forward(
    const gpu_t &gpu,
    fr_t *coeffs_inout,
    fr_t *scratch_split,
    uint32_t lg_domain_size,
    cpp_gl64_t shift)
{
    // `ntt::batch_ntt` rejects lg_domain_size == 0; do not compute n or launch kernels in that case.
    if (lg_domain_size == 0U || scratch_split == nullptr) {
        return RustError{cudaErrorInvalidValue};
    }

    const size_t n = (size_t)1U << lg_domain_size;
    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    fri_ext2_deinterleave<<<blocks, threads, 0, gpu>>>(coeffs_inout, scratch_split, n);
    CUDA_OK(cudaGetLastError());

    RustError ce = ntt::init_coset(gpu, lg_domain_size, fr_t(shift.get_val()));
    if (ce.code != 0) {
        return ce;
    }

    NTT_Config cfg = {};
    cfg.batches = 2;
    cfg.order = NN;
    cfg.ntt_type = standard;
    cfg.extension_rate_bits = 0;
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    cfg.with_coset = true;
    cfg.is_multi_gpu = false;
    cfg.salt_size = 0;

    RustError err = ntt::batch_ntt(gpu, (fr_t *)scratch_split, lg_domain_size, forward, cfg);
    if (err.code != 0) {
        return err;
    }

    fri_ext2_interleave<<<blocks, threads, 0, gpu>>>(scratch_split, coeffs_inout, n);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
    return RustError{cudaSuccess};
}

FriCommitPhaseResult::FriCommitPhaseResult()
    : final_coeffs_gpu(nullptr), final_num_ext_coeffs(0), gpu_id(0) {}

FriCommitPhaseResult::~FriCommitPhaseResult()
{
    clear_merkle_trees(trees);
    if (final_coeffs_gpu) {
        cudaFree(final_coeffs_gpu);
        final_coeffs_gpu = nullptr;
    }
}

FriCommitPhaseResult::FriCommitPhaseResult(FriCommitPhaseResult &&other) noexcept
    : trees(std::move(other.trees)),
      final_coeffs_gpu(other.final_coeffs_gpu),
      final_num_ext_coeffs(other.final_num_ext_coeffs),
      gpu_id(other.gpu_id)
{
    other.final_coeffs_gpu = nullptr;
    other.final_num_ext_coeffs = 0;
}

FriCommitPhaseResult &FriCommitPhaseResult::operator=(FriCommitPhaseResult &&other) noexcept
{
    if (this != &other) {
        clear_merkle_trees(trees);
        if (final_coeffs_gpu) {
            cudaFree(final_coeffs_gpu);
        }

        trees = std::move(other.trees);
        final_coeffs_gpu = other.final_coeffs_gpu;
        final_num_ext_coeffs = other.final_num_ext_coeffs;
        gpu_id = other.gpu_id;

        other.final_coeffs_gpu = nullptr;
        other.final_num_ext_coeffs = 0;
    }
    return *this;
}

void fri_commit_phase(
    const fr_t *lde_coeffs_gpu,
    const fr_t *lde_values_gpu,
    size_t num_ext_coeffs,
    const std::vector<size_t> &reduction_arity_bits,
    size_t rate_bits,
    size_t cap_height,
    Challenger &challenger,
    fr_t coset_shift,
    size_t gpu_id,
    FriCommitPhaseResult *out)
{
    if (!out) {
        throw cuda_error{-cudaErrorInvalidValue, "fri_commit_phase: out pointer is null"};
    }
    if (num_ext_coeffs == 0) {
        throw cuda_error{-cudaErrorInvalidValue, "fri_commit_phase: num_ext_coeffs is zero"};
    }
    size_t n = num_ext_coeffs;
    if ((n & (n - 1)) != 0) {
        throw cuda_error{-cudaErrorInvalidValue, "fri_commit_phase: num_ext_coeffs is not a power of two"};
    }
    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();

    uint32_t max_lg = (uint32_t)host_lg2(n);
    size_t nc = n;
    for (size_t ab : reduction_arity_bits) {
        size_t ar = (size_t)1 << ab;
        if (nc % ar != 0) {
            throw cuda_error{-cudaErrorInvalidValue,
                "fri_commit_phase: reduction arity does not divide domain size evenly"};
        }
        nc /= ar;
    }
    for (uint32_t lg = 2; lg <= max_lg; lg++) {
        ntt::init_twiddle_factors(gpu, lg);
    }

    cpp_gl64_t shift = coset_shift;

    fr_t *d_coeffs = nullptr;
    fr_t *d_values = nullptr;
    fr_t *d_folded = nullptr;
    fr_t *scratch_split = nullptr;
    fr_t *d_final = nullptr;
    const size_t ext_degree = 2;

    out->trees.clear();
    out->gpu_id = gpu_id;
    out->final_coeffs_gpu = nullptr;
    out->final_num_ext_coeffs = 0;

    try {
        size_t n_work = n;
        const size_t coeff_fr_count = n_work * ext_degree;
        size_t flat = coeff_fr_count * sizeof(fr_t);

        d_coeffs = cuda_malloc_fr_count(coeff_fr_count);
        d_values = cuda_malloc_fr_count(coeff_fr_count);
        CUDA_OK(cudaMemcpy(d_coeffs, lde_coeffs_gpu, flat, cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(d_values, lde_values_gpu, flat, cudaMemcpyDeviceToDevice));

        d_folded = cuda_malloc_fr_count(coeff_fr_count);

        size_t max_n = n_work;
        const size_t scratch_fr_count = 2 * max_n;
        scratch_split = cuda_malloc_fr_count(scratch_fr_count);

        std::vector<fr_t> host_cap;

        for (size_t round = 0; round < reduction_arity_bits.size(); round++) {
            size_t arity_bits = reduction_arity_bits[round];
            size_t arity = (size_t)1 << arity_bits;
            uint32_t lg_n = (uint32_t)host_lg2(n_work);

            MerkleTreeGPU tree{};
            try {
                tree.num_leaves = n_work / arity;
                tree.leaf_size = arity * ext_degree;
                tree.cap_len = (size_t)1 << cap_height;
                if (tree.cap_len > tree.num_leaves) {
                    tree.cap_len = tree.num_leaves;
                }
                tree.num_digests = 2 * (tree.num_leaves - tree.cap_len);

                tree.leaves_gpu = cuda_malloc_fr_count(tree.num_leaves * tree.leaf_size);
                launch_fri_prepare_merkle_leaves(
                    d_values, tree.leaves_gpu, n_work, lg_n, arity_bits, ext_degree, gpu);
                CUDA_OK(cudaGetLastError());

                size_t digests_alloc = (tree.num_digests == 0 ? NUM_HASH_OUT_ELTS
                                                              : tree.num_digests * NUM_HASH_OUT_ELTS);
                size_t cap_alloc = tree.cap_len * NUM_HASH_OUT_ELTS;
                tree.digests_gpu = cuda_malloc_fr_count(digests_alloc);
                tree.cap_gpu = cuda_malloc_fr_count(cap_alloc);

                fill_digests_buf_linear_gpu_with_gpu_ptr(
                    (void *)tree.digests_gpu,
                    (void *)tree.cap_gpu,
                    (void *)tree.leaves_gpu,
                    (u64)tree.num_digests,
                    (u64)tree.cap_len,
                    (u64)tree.num_leaves,
                    (u64)tree.leaf_size,
                    (u64)cap_height,
                    (u64)HashPoseidon,
                    (u64)gpu_id);

                gpu.sync();

                host_cap.resize(tree.cap_len * NUM_HASH_OUT_ELTS);
                if (!host_cap.empty()) {
                    CUDA_OK(cudaMemcpy(
                        host_cap.data(),
                        tree.cap_gpu,
                        host_cap.size() * sizeof(fr_t),
                        cudaMemcpyDeviceToHost));
                }
                challenger.observe_cap(host_cap.data(), host_cap.size());

                fr_t beta_host[2];
                challenger.get_extension_challenge(beta_host);

                fr_t *beta_dev = cuda_malloc_fr_count(2);
                CUDA_OK(cudaMemcpy(beta_dev, beta_host, 2 * sizeof(fr_t), cudaMemcpyHostToDevice));

                size_t folded_count = n_work / arity;
                launch_fri_fold_coefficients(
                    d_coeffs, d_folded, beta_dev, n_work, arity_bits, ext_degree, gpu);
                CUDA_OK(cudaGetLastError());
                cudaFree(beta_dev);

                std::swap(d_coeffs, d_folded);
                n_work = folded_count;

                shift = gl64_pow_u64(shift, (uint64_t)arity);

                if (n_work == 1) {
                    out->trees.push_back(tree);
                    tree.leaves_gpu = nullptr;
                    tree.digests_gpu = nullptr;
                    tree.cap_gpu = nullptr;
                    continue;
                }

                uint32_t lg_dom = (uint32_t)host_lg2(n_work);
                RustError fft_err = fri_ext2_coset_fft_forward(gpu, d_coeffs, scratch_split, lg_dom, shift);
                if (fft_err.code != 0) {
                    std::string msg = "fri_ext2_coset_fft_forward failed";
                    if (fft_err.message) {
                        msg += ": ";
                        msg += fft_err.message;
                    }
                    throw cuda_error{fft_err.code, msg};
                }

                CUDA_OK(cudaMemcpy(d_values, d_coeffs, n_work * ext_degree * sizeof(fr_t), cudaMemcpyDeviceToDevice));

                out->trees.push_back(tree);
                tree.leaves_gpu = nullptr;
                tree.digests_gpu = nullptr;
                tree.cap_gpu = nullptr;
            } catch (...) {
                free_merkle_tree(tree);
                throw;
            }
        }

        cudaFree(scratch_split);
        scratch_split = nullptr;
        cudaFree(d_folded);
        d_folded = nullptr;
        cudaFree(d_values);
        d_values = nullptr;

        size_t trunc = n_work >> rate_bits;
        if (trunc == 0) {
            trunc = 1;
        }
        if (trunc > n_work) {
            trunc = n_work;
        }

        d_final = cuda_malloc_fr_count(trunc * ext_degree);
        CUDA_OK(cudaMemcpy(d_final, d_coeffs, trunc * ext_degree * sizeof(fr_t), cudaMemcpyDeviceToDevice));
        cudaFree(d_coeffs);
        d_coeffs = nullptr;

        out->final_coeffs_gpu = d_final;
        out->final_num_ext_coeffs = trunc;
        d_final = nullptr;
    } catch (...) {
        if (d_final) {
            cudaFree(d_final);
        }
        if (d_coeffs) {
            cudaFree(d_coeffs);
        }
        if (d_values) {
            cudaFree(d_values);
        }
        if (d_folded) {
            cudaFree(d_folded);
        }
        if (scratch_split) {
            cudaFree(scratch_split);
        }
        clear_merkle_trees(out->trees);
        out->final_coeffs_gpu = nullptr;
        out->final_num_ext_coeffs = 0;
        throw;
    }
}

void fri_prepare_merkle_leaves_host_launch(
    const fr_t *d_values,
    fr_t *d_leaves,
    size_t n,
    size_t lg_n,
    size_t arity_bits,
    size_t ext_degree,
    size_t gpu_id)
{
    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();
    launch_fri_prepare_merkle_leaves(
        d_values, d_leaves, n, lg_n, arity_bits, ext_degree, gpu);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
}

void fri_fold_coefficients_host_launch(
    const fr_t *d_coeffs_in,
    fr_t *d_coeffs_out,
    const fr_t *d_beta,
    size_t n,
    size_t arity_bits,
    size_t ext_degree,
    size_t gpu_id)
{
    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();
    launch_fri_fold_coefficients(
        d_coeffs_in, d_coeffs_out, d_beta, n, arity_bits, ext_degree, gpu);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
}

#endif // !__CUDA_ARCH__
