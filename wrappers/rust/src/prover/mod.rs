// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! C layout mirrors `native/prover/types.h` and `native/prover/prover.h` (`gpu_prove`).

use core::ffi::c_void;

use crate::error;

pub const NUM_HASH_OUT_ELTS: usize = 4;

/// Magic `0x584e4b5a` (`ZKXN` LE) at the start of the GPU orchestrator proof blob (`prover.cu`).
pub const GPU_PROOF_MAGIC: u32 = 0x584e4b5a;

/// Conservative default buffer for [`gpu_prove_blob`]; resize if the API returns [`libc::E2BIG`].
pub const DEFAULT_GPU_PROOF_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// [`gpu_prove`] sets `*proof_size` to the required byte count when `RustError.code ==` this value
/// (POSIX `E2BIG`: proof output buffer too small).
pub const GPU_PROVE_ERR_PROOF_BUFFER_TOO_SMALL: i32 = libc::E2BIG;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProverConfig {
    pub degree_bits: u32,
    pub num_wires: u32,
    pub num_routed_wires: u32,
    pub num_challenges: u32,
    pub num_partial_products: u32,
    pub quotient_degree_factor: u32,
    pub rate_bits: u32,
    pub cap_height: u32,
    pub num_gate_constraints: u32,
    pub num_constants: u32,
    pub num_public_inputs: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateInfo {
    pub gate_type: u32,
    pub selector_index: u32,
    pub group_start: u32,
    pub group_end: u32,
    pub num_selectors: u32,
    pub num_constraints: u32,
    pub wire_0: u32,
    pub wire_1: u32,
    pub wire_2: u32,
    pub wire_3: u32,
    pub const_0: u32,
    pub const_1: u32,
    pub aux_0: u32,
    pub aux_1: u32,
}

extern "C" {
    pub fn gpu_prove(
        constants_sigmas_coeffs_gpu: *const c_void,
        constants_sigmas_lde_gpu: *const c_void,
        constants_sigmas_digests_gpu: *const c_void,
        constants_sigmas_cap_gpu: *const c_void,
        circuit_digest: *const c_void,
        public_inputs_hash: *const c_void,
        config: *const ProverConfig,
        gates: *const GateInfo,
        num_gates: u32,
        k_is: *const c_void,
        subgroup: *const c_void,
        wire_values: *const c_void,
        reduction_arity_bits: *const u32,
        num_fri_rounds: u32,
        fft_root_table: *const c_void,
        proof_output: *mut c_void,
        proof_size: *mut usize,
        gpu_id: u64,
    ) -> error::Error;
}

/// Host-side inputs for [`gpu_prove_blob`].
pub struct GpuProveBlobArgs<'a> {
    pub constants_sigmas_coeffs_gpu: *const c_void,
    pub constants_sigmas_lde_gpu: *const c_void,
    pub constants_sigmas_digests_gpu: *const c_void,
    pub constants_sigmas_cap_gpu: *const c_void,
    pub circuit_digest_gl64: &'a [u64; NUM_HASH_OUT_ELTS],
    pub public_inputs_hash_gl64: &'a [u64; NUM_HASH_OUT_ELTS],
    pub config: &'a ProverConfig,
    pub gates: &'a [GateInfo],
    pub k_is_gl64: &'a [u64],
    pub subgroup_gl64: &'a [u64],
    pub wire_values_gl64: &'a [u64],
    pub reduction_arity_bits: Option<&'a [u32]>,
    pub num_fri_rounds: u32,
    pub fft_root_table: *const c_void,
    pub gpu_id: u64,
}

/// Validates slice lengths, allocates a buffer (up to `max_bytes`), and invokes [`gpu_prove`].
///
/// On [`GPU_PROVE_ERR_PROOF_BUFFER_TOO_SMALL`] ([`libc::E2BIG`]), the native prover sets `*proof_size`
/// to the required byte count; this function grows the buffer once and retries.
pub fn gpu_prove_blob(args: GpuProveBlobArgs<'_>, max_bytes: usize) -> Result<Vec<u8>, String> {
    let degree = 1usize
        .checked_shl(args.config.degree_bits)
        .ok_or_else(|| "gpu_prove_blob: degree_bits too large".to_string())?;
    let nw = args.config.num_wires as usize;
    let nr = args.config.num_routed_wires as usize;

    if args.k_is_gl64.len() != nr {
        return Err(format!(
            "gpu_prove_blob: k_is length {} != num_routed_wires {}",
            args.k_is_gl64.len(),
            nr
        ));
    }
    if args.subgroup_gl64.len() != degree {
        return Err(format!(
            "gpu_prove_blob: subgroup length {} != degree {}",
            args.subgroup_gl64.len(),
            degree
        ));
    }
    if args.wire_values_gl64.len() != nw * degree {
        return Err(format!(
            "gpu_prove_blob: wire_values length {} != num_wires*degree {}",
            args.wire_values_gl64.len(),
            nw * degree
        ));
    }
    if args.constants_sigmas_coeffs_gpu.is_null() {
        return Err("gpu_prove_blob: constants_sigmas_coeffs_gpu is null".into());
    }

    let gates_ptr = if args.gates.is_empty() {
        core::ptr::null()
    } else {
        args.gates.as_ptr()
    };
    let num_gates =
        u32::try_from(args.gates.len()).map_err(|_| "gpu_prove_blob: too many gates")?;

    let arity_ptr = args
        .reduction_arity_bits
        .map(|s| s.as_ptr())
        .unwrap_or(core::ptr::null());

    let mut buf = vec![0u8; DEFAULT_GPU_PROOF_BUFFER_BYTES.min(max_bytes).max(4096)];
    let mut proof_size = buf.len();

    for attempt in 0..2 {
        let err = unsafe {
            gpu_prove(
                args.constants_sigmas_coeffs_gpu,
                args.constants_sigmas_lde_gpu,
                args.constants_sigmas_digests_gpu,
                args.constants_sigmas_cap_gpu,
                args.circuit_digest_gl64.as_ptr() as *const c_void,
                args.public_inputs_hash_gl64.as_ptr() as *const c_void,
                args.config,
                gates_ptr,
                num_gates,
                args.k_is_gl64.as_ptr() as *const c_void,
                args.subgroup_gl64.as_ptr() as *const c_void,
                args.wire_values_gl64.as_ptr() as *const c_void,
                arity_ptr,
                args.num_fri_rounds,
                args.fft_root_table,
                buf.as_mut_ptr() as *mut c_void,
                &mut proof_size,
                args.gpu_id,
            )
        };

        if err.code == 0 {
            buf.truncate(proof_size);
            return Ok(buf);
        }

        let msg = String::from(&err);
        if attempt == 0 && err.code == libc::E2BIG && proof_size > buf.len() {
            if proof_size > max_bytes {
                return Err(format!(
                    "gpu_prove_blob: need {} bytes (max {})",
                    proof_size, max_bytes
                ));
            }
            buf.resize(proof_size, 0);
            continue;
        }
        return Err(msg);
    }
    Err("gpu_prove_blob: retry exhausted".into())
}

#[cfg(feature = "plonky2")]
mod plonky2;

#[cfg(feature = "plonky2")]
mod zkxn_blob;

mod quotient_polys;

#[cfg(feature = "plonky2")]
pub use plonky2::{gate_type_id, gpu_prove_safe, plonky2_gate_infos};
pub use quotient_polys::compute_quotient_polys_device_gl64;

#[cfg(feature = "plonky2")]
pub use zkxn_blob::{
    parse_zkxn_gpu_blob, proof_with_public_inputs_from_zkxn_blob,
    verified_proof_matching_zkxn_blob, zkxn_opening_limbs_for_gpu_prover_config,
    zkxn_opening_limbs_for_plonky2_common, ParsedZkxnBlob, ZkxnBlobError,
};
