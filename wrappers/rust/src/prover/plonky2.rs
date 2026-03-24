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
    let v = h.to_vec();
    if v.len() != NUM_HASH_OUT_ELTS {
        return Err(format!(
            "GPU prover expects {} Goldilocks limbs in the hash; got {}",
            NUM_HASH_OUT_ELTS,
            v.len()
        ));
    }
    Ok([
        v[0].to_canonical_u64(),
        v[1].to_canonical_u64(),
        v[2].to_canonical_u64(),
        v[3].to_canonical_u64(),
    ])
}

/// Serializes Plonky2 circuit/witness data and runs [`super::gpu_prove_blob`].
///
/// Returns the **native GPU proof blob** (magic [`super::GPU_PROOF_MAGIC`]), not `Proof::to_bytes`.
/// Converting this into [`plonky2::plonk::proof::ProofWithPublicInputs`] requires matching FRI and
/// transcript wiring on the C++ side and is not implemented here.
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
    let mut wire_values_flat: Vec<u64> = Vec::with_capacity(nw * degree);
    for w in 0..nw {
        for r in 0..degree {
            wire_values_flat.push(witness.get_wire(r, w).to_canonical_u64());
        }
    }

    let mut k_is: Vec<u64> = Vec::with_capacity(common_data.k_is.len());
    for x in &common_data.k_is {
        k_is.push(x.to_canonical_u64());
    }

    let mut subgroup: Vec<u64> = Vec::with_capacity(prover_data.subgroup.len());
    for x in &prover_data.subgroup {
        subgroup.push(x.to_canonical_u64());
    }

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

#[cfg(all(test, feature = "plonky2", feature = "cuda", not(feature = "no_cuda")))]
mod fibonacci_gpu_tests {
    use core::ffi::c_void;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use plonky2::iop::generator::generate_partial_witness;
    use plonky2::iop::witness::{PartialWitness, Witness, WitnessWrite};
    use plonky2::plonk::circuit_builder::CircuitBuilder;
    use plonky2::plonk::circuit_data::CircuitConfig;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use plonky2::plonk::prover::prove;
    use plonky2::plonk::verifier::verify;
    use plonky2::util::timing::TimingTree;
    use rustacuda::memory::DeviceBuffer;
    use rustacuda::quick_init;

    use super::gpu_prove_safe;
    use crate::prover::GPU_PROOF_MAGIC;
    use crate::{get_number_of_gpus_rs, init_cuda_degree_rs, init_cuda_rs};

    /// Same iteration count as `plonky2/plonky2/examples/fibonacci.rs` (100th Fibonacci element).
    const FIB_ADD_STEPS: usize = 99;

    fn build_fibonacci_circuit<C: GenericConfig<2, F = GoldilocksField>>() -> (
        plonky2::plonk::circuit_data::CircuitData<GoldilocksField, C, 2>,
        PartialWitness<GoldilocksField>,
    )
    where
        C::Hasher: plonky2::plonk::config::Hasher<GoldilocksField>,
    {
        const D: usize = 2;
        type F = GoldilocksField;
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();
        let mut prev_target = initial_a;
        let mut cur_target = initial_b;
        for _ in 0..FIB_ADD_STEPS {
            let temp = builder.add(prev_target, cur_target);
            prev_target = cur_target;
            cur_target = temp;
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(cur_target);

        let mut pw = PartialWitness::new();
        pw.set_target(initial_a, F::ZERO);
        pw.set_target(initial_b, F::ONE);

        let data = builder.build::<C>();
        (data, pw)
    }

    fn flatten_constants_sigmas_coeffs(
        commitment: &plonky2::fri::oracle::PolynomialBatch<
            GoldilocksField,
            PoseidonGoldilocksConfig,
            2,
        >,
    ) -> Vec<u64> {
        let degree = commitment.polynomials[0].coeffs.len();
        let n_polys = commitment.polynomials.len();
        let mut flat = Vec::with_capacity(degree * n_polys);
        for p in &commitment.polynomials {
            assert_eq!(p.coeffs.len(), degree);
            for c in &p.coeffs {
                flat.push(c.to_canonical_u64());
            }
        }
        flat
    }

    /// End-to-end Fibonacci (example circuit) with Zeknox `gpu_prove` and Plonky2 verification.
    ///
    /// The native GPU orchestrator returns a **custom proof blob** (Merkle caps + openings) and does
    /// not yet emit a full Plonky2 [`plonky2::plonk::proof::Proof`] (no FRI in the blob). Therefore
    /// [`verify`] cannot consume the GPU bytes directly. This test:
    /// 1. Runs Plonky2 [`prove`] + [`verify`] for the Fibonacci-style circuit (statement correctness).
    /// 2. Runs [`gpu_prove_safe`] on the **same** witness and checks the blob header (GPU path).
    #[test]
    fn fibonacci_gpu_blob_and_plonky2_verify() {
        type C = PoseidonGoldilocksConfig;

        assert!(
            get_number_of_gpus_rs().unwrap_or(0) >= 1,
            "need at least one CUDA device for gpu_prove_safe"
        );

        let (data, pw) = build_fibonacci_circuit::<C>();

        let mut timing = TimingTree::default();
        let proof =
            prove(&data.prover_only, &data.common, pw.clone(), &mut timing).expect("cpu prove");
        let public_inputs_ref = proof.public_inputs.clone();
        verify(proof, &data.verifier_only, &data.common).expect("plonky2 verify");

        // `init_cuda_degree` is a *log-domain* bound (see `native/lib.cu`: loops `k = 2..max_degree`).
        init_cuda_rs();
        let max_lg_domain = data.common.degree_bits() + data.common.config.fri_config.rate_bits + 4;
        init_cuda_degree_rs(max_lg_domain);

        let partition = generate_partial_witness(pw, &data.prover_only, &data.common);
        let public_inputs = partition.get_targets(&data.prover_only.public_inputs);
        assert_eq!(public_inputs, public_inputs_ref);
        let witness = partition.full_witness();

        let flat = flatten_constants_sigmas_coeffs(&data.prover_only.constants_sigmas_commitment);
        let _rusta = quick_init().expect("CUDA context for coeff upload");
        let mut d_coeffs = DeviceBuffer::from_slice(&flat).expect("device alloc coeffs");
        let coeffs_dev = (&mut *d_coeffs).as_device_ptr().as_raw() as *const c_void;

        let blob = gpu_prove_safe(
            &data.prover_only,
            &data.common,
            &witness,
            &public_inputs,
            coeffs_dev,
            0,
        )
        .expect("gpu_prove_safe");
        assert!(blob.len() >= 4);
        assert_eq!(
            u32::from_le_bytes(blob[0..4].try_into().unwrap()),
            GPU_PROOF_MAGIC
        );
    }
}
