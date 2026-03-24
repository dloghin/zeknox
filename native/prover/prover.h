// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_PROVER_H__
#define __ZEKNOX_PROVER_PROVER_H__

#include <stddef.h>
#include <stdint.h>

#include <utils/rusterror.h>

#include <prover/types.h>

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

/**
 * Top-level GPU prover: wires → commitments → partial products → quotient batch → opening set → proof blob.
 *
 * Witness layout: column-major `[num_wires × degree]` with `degree = 2^degree_bits`, i.e. wire `w` at trace
 * row `r` is at index `w * degree + r` (matches `PolynomialBatchGPU::from_values` and Plonky2 trace order).
 *
 * Partial products use row-major trace layout internally (`row * num_wires + wire`); the implementation
 * transposes a copy for the permutation-argument kernel.
 *
 * Precomputed `constants_sigmas_*` pointers may alias the same buffers used to build `PolynomialBatchGPU`
 * via `from_coeffs` (coefficients on device). LDE/digest fields are optional for future fast paths; the
 * orchestrator currently commits from coefficients only.
 *
 * If `num_fri_rounds == 0`, FRI is skipped and the proof contains Merkle caps plus flattened openings only.
 * If `num_fri_rounds > 0`, the call fails until composition + FRI wiring is completed.
 *
 * If `proof_output` is too small, returns `RustError.code == E2BIG` (POSIX errno) and sets `*proof_size`
 * to the required output size in bytes.
 */
EXTERN RustError gpu_prove(
    const void *constants_sigmas_coeffs_gpu,
    const void *constants_sigmas_lde_gpu,
    const void *constants_sigmas_digests_gpu,
    const void *constants_sigmas_cap_gpu,

    const void *circuit_digest,
    const void *public_inputs_hash,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const void *k_is,
    const void *subgroup,

    const void *wire_values,

    const uint32_t *reduction_arity_bits,
    uint32_t num_fri_rounds,
    const void *fft_root_table,

    void *proof_output,
    size_t *proof_size,

    uint64_t gpu_id);

#undef EXTERN

#endif // __ZEKNOX_PROVER_PROVER_H__
