// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_QUOTIENT_COMPUTE_POLYS_CUH__
#define __ZEKNOX_PROVER_QUOTIENT_COMPUTE_POLYS_CUH__

#include <stddef.h>
#include <stdint.h>

#include <utils/rusterror.h>

#include <prover/types.h>

#ifdef __cplusplus
#include <cuda_runtime.h>
#include <prover/polynomial_batch.cuh>
#include <prover/gate_constraints.cuh>
#endif

/**
 * Plonky2 `compute_quotient_polys` (prover.rs): evaluate the vanishing polynomial on the
 * subsampled quotient LDE domain (size 2^(degree_bits + log2_ceil(quotient_degree_factor))),
 * divide by Z_H on the coset, then inverse coset NTT per challenge.
 *
 * Expects `constants_sigmas`, `wires`, and `zs_partial_products` batches already built with
 * `PolynomialBatchGPU::from_values` / `from_coeffs` (same LDE layout as Plonky2 `PolynomialBatch`).
 *
 * @param d_out_quotient_coeffs  Device buffer, size `num_challenges * lde_q_size` (`fr_t` / u64).
 *                               Layout: challenge-major, then coefficient index (coset IFFT output).
 * @param out_lde_q_size         Host output: `lde_q_size = 2^(degree_bits + quotient_degree_bits)`.
 */
#ifdef __cplusplus
RustError compute_quotient_polys_from_batches_gl64(
    size_t gpu_id,
    cudaStream_t stream,
    const PolynomialBatchGPU &constants_sigmas,
    const PolynomialBatchGPU &wires,
    const PolynomialBatchGPU &zs_partial_products,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const uint64_t *h_k_is,
    const uint64_t *h_public_inputs_hash,
    const uint64_t *h_betas,
    const uint64_t *h_gammas,
    const uint64_t *h_alphas,
    fr_t *d_out_quotient_coeffs,
    size_t *out_lde_q_size);
#endif

#endif // __ZEKNOX_PROVER_QUOTIENT_COMPUTE_POLYS_CUH__
