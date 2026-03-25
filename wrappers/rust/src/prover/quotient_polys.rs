// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Device-side Plonky2 [`compute_quotient_polys`](https://github.com/mir-protocol/plonky2/blob/main/plonky2/src/plonk/prover.rs)
//! (vanishing polynomial on the subsampled quotient LDE, divide by `Z_H`, inverse coset NTT per challenge).
//!
//! All `d_*` pointers must be **device** pointers in the same LDE layout as native [`PolynomialBatchGPU`]
//! (bit-reversed coset LDE leaves). For `num_challenges == 1`, [`super::gpu_prove`] uses this path internally.

use core::ffi::c_void;

use crate::error;
use crate::prover::{GateInfo, ProverConfig};

extern "C" {
    fn zeknox_compute_quotient_polys_gl64(
        gpu_id: usize,
        cuda_stream: *mut c_void,
        d_constants_sigmas_lde: *const u64,
        constants_sigmas_leaf_size: u64,
        d_wires_lde: *const u64,
        wires_leaf_size: u64,
        d_zs_partial_products_lde: *const u64,
        zs_partial_leaf_size: u64,
        config: *const ProverConfig,
        gates: *const GateInfo,
        num_gates: u32,
        h_k_is: *const u64,
        h_public_inputs_hash: *const u64,
        h_betas: *const u64,
        h_gammas: *const u64,
        h_alphas: *const u64,
        d_out_quotient_coeffs: *mut u64,
        out_lde_q_size: *mut usize,
    ) -> error::Error;
}

/// Runs the native Plonky2 quotient pipeline on GPU.
///
/// # Safety
/// Caller must pass valid CUDA device pointers and a valid `cuda_stream` (`cudaStream_t` as `*mut c_void`).
/// Output buffer must hold at least `config.num_challenges * out_lde_q_size` Goldilocks elements (written on success).
pub unsafe fn compute_quotient_polys_device_gl64(
    gpu_id: usize,
    cuda_stream: *mut c_void,
    d_constants_sigmas_lde: *const u64,
    constants_sigmas_leaf_size: u64,
    d_wires_lde: *const u64,
    wires_leaf_size: u64,
    d_zs_partial_products_lde: *const u64,
    zs_partial_leaf_size: u64,
    config: *const ProverConfig,
    gates: *const GateInfo,
    num_gates: u32,
    h_k_is: *const u64,
    h_public_inputs_hash: *const u64,
    h_betas: *const u64,
    h_gammas: *const u64,
    h_alphas: *const u64,
    d_out_quotient_coeffs: *mut u64,
    out_lde_q_size: *mut usize,
) -> Result<(), error::Error> {
    let err = zeknox_compute_quotient_polys_gl64(
        gpu_id,
        cuda_stream,
        d_constants_sigmas_lde,
        constants_sigmas_leaf_size,
        d_wires_lde,
        wires_leaf_size,
        d_zs_partial_products_lde,
        zs_partial_leaf_size,
        config,
        gates,
        num_gates,
        h_k_is,
        h_public_inputs_hash,
        h_betas,
        h_gammas,
        h_alphas,
        d_out_quotient_coeffs,
        out_lde_q_size,
    );
    if err.code == 0 {
        Ok(())
    } else {
        Err(err)
    }
}
