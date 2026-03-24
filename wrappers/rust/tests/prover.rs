// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Prover FFI layout checks and Plonky2 + `gpu_prove_safe` integration.

#[cfg(all(feature = "cuda", not(feature = "no_cuda")))]
mod ffi_layout {
    use zeknox::{GateInfo, ProverConfig};

    #[test]
    fn prover_config_matches_native_layout() {
        assert_eq!(std::mem::size_of::<ProverConfig>(), 11 * 4);
    }

    #[test]
    fn gate_info_matches_native_layout() {
        assert_eq!(std::mem::size_of::<GateInfo>(), 6 * 4);
    }
}

#[cfg(all(feature = "plonky2", feature = "cuda", not(feature = "no_cuda")))]
mod fibonacci {
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

    use zeknox::{
        get_number_of_gpus_rs, gpu_prove_safe, init_cuda_degree_rs, init_cuda_rs, GPU_PROOF_MAGIC,
    };

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
