// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::ffi::c_void;

use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::PrimeField64;
use plonky2::gates::gate::GateRef;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::MatrixWitness;
use plonky2::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use plonky2::plonk::config::{GenericConfig, GenericHashOut, Hasher};

use super::{
    gpu_prove_blob, GateInfo, GpuProveBlobArgs, ProverConfig, DEFAULT_GPU_PROOF_BUFFER_BYTES,
    NUM_HASH_OUT_ELTS,
};

/// Maps Plonky2 gate type names to the native prover enum (metadata for future gate wiring).
pub fn gate_type_id<F: RichField + Extendable<D>, const D: usize>(gate: &GateRef<F, D>) -> u32 {
    let id = gate.0.id();
    let head = id
        .split(|c| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or(id.as_str());
    match head {
        "ArithmeticGate" => 0,
        "ArithmeticExtensionGate" => 1,
        "ConstantGate" => 2,
        "PublicInputGate" => 3,
        "PoseidonGate" => 4,
        "BaseSumGate" => 5,
        "RandomAccessGate" => 6,
        "NoopGate" => 7,
        _ => panic!("Unsupported gate type for GPU prover metadata: {}", id),
    }
}

/// One [`GateInfo`] per entry in [`CommonCircuitData::gates`], aligned with selector groups when possible.
pub fn plonky2_gate_infos<F: RichField + Extendable<D>, const D: usize>(
    common: &CommonCircuitData<F, D>,
) -> Vec<GateInfo> {
    let num_sel = common.selectors_info.num_selectors() as u32;
    common
        .gates
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let sel = common
                .selectors_info
                .selector_indices
                .get(i)
                .copied()
                .unwrap_or(0) as u32;
            let (gs, ge) = common
                .selectors_info
                .groups
                .get(i)
                .map(|r| (r.start as u32, r.end as u32))
                .unwrap_or((0, 0));
            GateInfo {
                gate_type: gate_type_id(g),
                selector_index: sel,
                group_start: gs,
                group_end: ge,
                num_selectors: num_sel,
                num_constraints: g.0.num_constraints() as u32,
            }
        })
        .collect()
}

fn hash_out_to_digest<H: GenericHashOut<F>, F: RichField>(
    h: &H,
) -> Result<[u64; NUM_HASH_OUT_ELTS], String> {
    let arr: [F; NUM_HASH_OUT_ELTS] = h.to_vec().try_into().map_err(|v: Vec<F>| {
        format!(
            "GPU prover expects {} Goldilocks limbs in the hash; got {}",
            NUM_HASH_OUT_ELTS,
            v.len()
        )
    })?;
    Ok(arr.map(|f| f.to_canonical_u64()))
}

/// Serializes Plonky2 circuit/witness data and runs [`super::gpu_prove_blob`].
///
/// Returns the **native GPU proof blob** (magic [`super::GPU_PROOF_MAGIC`]), not `Proof::to_bytes`.
/// To build a Plonky2 [`plonky2::plonk::proof::ProofWithPublicInputs`] for
/// [`plonky2::plonk::verifier::verify`], use [`super::proof_with_public_inputs_from_zkxn_blob`]
/// together with a transcript-consistent [`plonky2::fri::proof::FriProof`] (typically from the CPU
/// prover). That path succeeds only when GPU opening layout matches Plonky2’s [`plonky2::plonk::proof::OpeningSet`] shape
/// (see [`super::zkxn_opening_limbs_for_gpu_prover_config`] vs
/// [`super::zkxn_opening_limbs_for_plonky2_common`]) and the native quotient matches Plonky2.
///
/// `constants_sigmas_coeffs_gpu` must be a device pointer to coefficient-domain polynomials in the
/// same layout as [`plonky2::fri::oracle::PolynomialBatch`] / native `PolynomialBatchGPU::from_coeffs`
/// (column-major, `(num_constants + num_routed_wires) * degree` base-field elements).
pub fn gpu_prove_safe<C: GenericConfig<2, F = GoldilocksField>>(
    prover_data: &ProverOnlyCircuitData<GoldilocksField, C, 2>,
    common_data: &CommonCircuitData<GoldilocksField, 2>,
    witness: &MatrixWitness<GoldilocksField>,
    public_inputs: &[GoldilocksField],
    constants_sigmas_coeffs_gpu: *const c_void,
    gpu_id: usize,
) -> Result<Vec<u8>, String>
where
    C::Hasher: Hasher<GoldilocksField>,
{
    let config = ProverConfig {
        degree_bits: common_data.degree_bits() as u32,
        num_wires: common_data.config.num_wires as u32,
        num_routed_wires: common_data.config.num_routed_wires as u32,
        num_challenges: common_data.config.num_challenges as u32,
        num_partial_products: common_data.num_partial_products as u32,
        quotient_degree_factor: common_data.quotient_degree_factor as u32,
        rate_bits: common_data.config.fri_config.rate_bits as u32,
        cap_height: common_data.config.fri_config.cap_height as u32,
        num_gate_constraints: common_data.num_gate_constraints as u32,
        num_constants: common_data.num_constants as u32,
        num_public_inputs: common_data.num_public_inputs as u32,
    };

    let gates = plonky2_gate_infos(common_data);

    let degree = common_data.degree();
    let nw = common_data.config.num_wires;

    let wire_values_flat: Vec<u64> = (0..nw)
        .flat_map(|w| (0..degree).map(move |r| witness.get_wire(r, w).to_canonical_u64()))
        .collect();

    let k_is: Vec<u64> = common_data
        .k_is
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    let subgroup: Vec<u64> = prover_data
        .subgroup
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    let circuit_digest = hash_out_to_digest(&prover_data.circuit_digest)?;
    let public_inputs_hash =
        hash_out_to_digest(&C::InnerHasher::hash_public_inputs(public_inputs))?;

    // Native `gpu_prove` rejects `num_fri_rounds > 0` until FRI is wired; keep arity table unset for MVP.
    let args = GpuProveBlobArgs {
        constants_sigmas_coeffs_gpu,
        constants_sigmas_lde_gpu: core::ptr::null(),
        constants_sigmas_digests_gpu: core::ptr::null(),
        constants_sigmas_cap_gpu: core::ptr::null(),
        circuit_digest_gl64: &circuit_digest,
        public_inputs_hash_gl64: &public_inputs_hash,
        config: &config,
        gates: &gates,
        k_is_gl64: &k_is,
        subgroup_gl64: &subgroup,
        wire_values_gl64: &wire_values_flat,
        reduction_arity_bits: None,
        num_fri_rounds: 0,
        fft_root_table: core::ptr::null(),
        gpu_id: gpu_id as u64,
    };

    gpu_prove_blob(args, DEFAULT_GPU_PROOF_BUFFER_BYTES * 4)
}
