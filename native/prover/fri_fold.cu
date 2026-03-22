// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cmath>
#include <cstring>

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
    if (lg_domain_size == 0) {
        return RustError{cudaErrorInvalidValue};
    }
    size_t n = (size_t)1 << lg_domain_size;
    int threads = 256;
    int blocks = (int)((n + (size_t)threads - 1) / (size_t)threads);
    fri_ext2_deinterleave<<<blocks, threads, 0, gpu>>>(coeffs_inout, scratch_split, n);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaPeekAtLastError());

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
    for (auto &t : trees) {
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
    trees.clear();
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
        this->~FriCommitPhaseResult();
        new (this) FriCommitPhaseResult(std::move(other));
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
    if (!out || num_ext_coeffs == 0) {
        return;
    }
    size_t n = num_ext_coeffs;
    if ((n & (n - 1)) != 0) {
        return;
    }
    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();

    uint32_t max_lg = (uint32_t)host_lg2(n);
    size_t nc = n;
    for (size_t ab : reduction_arity_bits) {
        size_t ar = (size_t)1 << ab;
        if (nc % ar != 0) {
            return;
        }
        nc /= ar;
    }
    for (uint32_t lg = 2; lg <= max_lg; lg++) {
        ntt::init_twiddle_factors(gpu, lg);
    }

    cpp_gl64_t shift = coset_shift;

    fr_t *d_coeffs = nullptr;
    fr_t *d_values = nullptr;
    size_t ext_degree = 2;
    size_t flat = n * ext_degree * sizeof(fr_t);
    CUDA_OK(cudaMalloc(&d_coeffs, flat));
    CUDA_OK(cudaMalloc(&d_values, flat));
    CUDA_OK(cudaMemcpy(d_coeffs, lde_coeffs_gpu, flat, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(d_values, lde_values_gpu, flat, cudaMemcpyDeviceToDevice));

    fr_t *d_folded = nullptr;
    CUDA_OK(cudaMalloc(&d_folded, flat));

    size_t max_n = n;
    fr_t *scratch_split = nullptr;
    CUDA_OK(cudaMalloc(&scratch_split, 2 * max_n * sizeof(fr_t)));

    std::vector<fr_t> host_cap;

    out->trees.clear();
    out->gpu_id = gpu_id;

    for (size_t round = 0; round < reduction_arity_bits.size(); round++) {
        size_t arity_bits = reduction_arity_bits[round];
        size_t arity = (size_t)1 << arity_bits;
        uint32_t lg_n = (uint32_t)host_lg2(n);

        MerkleTreeGPU tree{};
        tree.num_leaves = n / arity;
        tree.leaf_size = arity * ext_degree;
        tree.cap_len = (size_t)1 << cap_height;
        if (tree.cap_len > tree.num_leaves) {
            tree.cap_len = tree.num_leaves;
        }
        tree.num_digests = 2 * (tree.num_leaves - tree.cap_len);

        CUDA_OK(cudaMalloc(&tree.leaves_gpu, tree.num_leaves * tree.leaf_size * sizeof(fr_t)));
        launch_fri_prepare_merkle_leaves(
            d_values, tree.leaves_gpu, n, lg_n, arity_bits, ext_degree, gpu);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaPeekAtLastError());

        size_t digests_alloc = (tree.num_digests == 0 ? NUM_HASH_OUT_ELTS
                                                      : tree.num_digests * NUM_HASH_OUT_ELTS);
        size_t cap_alloc = tree.cap_len * NUM_HASH_OUT_ELTS;
        CUDA_OK(cudaMalloc(&tree.digests_gpu, digests_alloc * sizeof(fr_t)));
        CUDA_OK(cudaMalloc(&tree.cap_gpu, cap_alloc * sizeof(fr_t)));

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

        fr_t *beta_dev = nullptr;
        CUDA_OK(cudaMalloc(&beta_dev, 2 * sizeof(fr_t)));
        CUDA_OK(cudaMemcpy(beta_dev, beta_host, 2 * sizeof(fr_t), cudaMemcpyHostToDevice));

        size_t folded_count = n / arity;
        launch_fri_fold_coefficients(
            d_coeffs, d_folded, beta_dev, n, arity_bits, ext_degree, gpu);
        CUDA_OK(cudaGetLastError());
        cudaFree(beta_dev);

        std::swap(d_coeffs, d_folded);
        n = folded_count;

        shift = gl64_pow_u64(shift, (uint64_t)arity);

        if (n == 1) {
            CUDA_OK(cudaMemcpy(d_values, d_coeffs, ext_degree * sizeof(fr_t), cudaMemcpyDeviceToDevice));
            out->trees.push_back(tree);
            continue;
        }

        uint32_t lg_dom = (uint32_t)host_lg2(n);
        RustError fft_err = fri_ext2_coset_fft_forward(gpu, d_coeffs, scratch_split, lg_dom, shift);
        if (fft_err.code != 0) {
            for (auto &t : out->trees) {
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
            out->trees.clear();
            cudaFree(scratch_split);
            cudaFree(d_folded);
            cudaFree(d_values);
            cudaFree(d_coeffs);
            return;
        }

        CUDA_OK(cudaMemcpy(d_values, d_coeffs, n * ext_degree * sizeof(fr_t), cudaMemcpyDeviceToDevice));

        out->trees.push_back(tree);
    }

    cudaFree(scratch_split);
    cudaFree(d_folded);
    cudaFree(d_values);

    size_t trunc = n >> rate_bits;
    if (trunc == 0) {
        trunc = 1;
    }
    if (trunc > n) {
        trunc = n;
    }

    fr_t *d_final = nullptr;
    CUDA_OK(cudaMalloc(&d_final, trunc * ext_degree * sizeof(fr_t)));
    CUDA_OK(cudaMemcpy(d_final, d_coeffs, trunc * ext_degree * sizeof(fr_t), cudaMemcpyDeviceToDevice));
    cudaFree(d_coeffs);

    out->final_coeffs_gpu = d_final;
    out->final_num_ext_coeffs = trunc;
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
