// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_FRI_FOLD_CUH__
#define __ZEKNOX_PROVER_FRI_FOLD_CUH__

#include <cstddef>
#include <cstdint>
#include <vector>

#include <ff/goldilocks.hpp>

class Challenger;

/// One FRI round: prepared leaves + Merkle digests/cap (same layout as `PolynomialBatchGPU` Merkle).
struct MerkleTreeGPU {
    fr_t *digests_gpu;
    fr_t *cap_gpu;
    fr_t *leaves_gpu;
    size_t num_leaves;
    size_t leaf_size;
    size_t num_digests;
    size_t cap_len;
};

struct FriCommitPhaseResult {
    std::vector<MerkleTreeGPU> trees;
    /// Device pointer to `final_num_ext_coeffs` quadratic extension coefficients (2 * count base limbs).
    fr_t *final_coeffs_gpu;
    size_t final_num_ext_coeffs;
    size_t gpu_id;

    FriCommitPhaseResult();
    ~FriCommitPhaseResult();

    FriCommitPhaseResult(const FriCommitPhaseResult &) = delete;
    FriCommitPhaseResult &operator=(const FriCommitPhaseResult &) = delete;

    FriCommitPhaseResult(FriCommitPhaseResult &&other) noexcept;
    FriCommitPhaseResult &operator=(FriCommitPhaseResult &&other) noexcept;
};

/// Host: run FRI commit rounds (Merkle + challenger + fold + coset FFT on extension coeffs).
/// `num_ext_coeffs` is the number of `gl64_ext2_t` coefficients (domain length); must be a power of two.
/// `coset_shift` is the initial base-field coset generator (e.g. Plonky2 `coset_shift`, often 7).
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
    FriCommitPhaseResult *out);

/// Test helpers: launch kernels on `gpu_id` default stream (sync before return).
void fri_prepare_merkle_leaves_host_launch(
    const fr_t *d_values,
    fr_t *d_leaves,
    size_t n,
    size_t lg_n,
    size_t arity_bits,
    size_t ext_degree,
    size_t gpu_id);

void fri_fold_coefficients_host_launch(
    const fr_t *d_coeffs_in,
    fr_t *d_coeffs_out,
    const fr_t *d_beta,
    size_t n,
    size_t arity_bits,
    size_t ext_degree,
    size_t gpu_id);

#endif // __ZEKNOX_PROVER_FRI_FOLD_CUH__
