# Task 10: Rust FFI Wrapper

## Objective

Add the `gpu_prove` extern declaration and safe Rust wrapper to `wrappers/rust/src/lib.rs`, including serialization of Plonky2 circuit data to C structs, enabling the Plonky2 Rust prover to call the CUDA prover.

## Background

The existing Rust FFI wrappers in `wrappers/rust/src/lib.rs` expose functions like `lde_batch`, `ntt_batch`, `fill_digests_buf_linear_gpu_with_gpu_ptr` via `extern "C"` blocks. The same pattern will be used for the prover.

The integration point is in `plonky2/plonky2/src/plonk/prover.rs`, where a `#[cfg(feature = "cuda")]` conditional can dispatch to the GPU prover.

## Deliverables

### 1. FFI Declaration in `wrappers/rust/src/lib.rs`

```rust
#[repr(C)]
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
pub struct GateInfo {
    pub gate_type: u32,
    pub selector_index: u32,
    pub group_start: u32,
    pub group_end: u32,
    pub num_selectors: u32,
    pub num_constraints: u32,
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
```

### 2. Safe Wrapper Function

```rust
pub fn gpu_prove_safe<F, C, D>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    witness: &MatrixWitness<F>,
    public_inputs: &[F],
    gpu_id: usize,
) -> Result<ProofWithPublicInputs<F, C, D>, String>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    C::Hasher: Hasher<F>,
{
    // 1. Serialize ProverConfig from common_data
    let config = ProverConfig {
        degree_bits: common_data.degree_bits() as u32,
        num_wires: common_data.config.num_wires as u32,
        // ... etc
    };

    // 2. Serialize gate info
    let gates: Vec<GateInfo> = common_data.gates.iter().enumerate().map(|(i, gate)| {
        GateInfo {
            gate_type: gate_type_id(gate),  // map gate to enum
            // ...
        }
    }).collect();

    // 3. Flatten wire values to column-major layout
    let wire_values_flat: Vec<F> = (0..common_data.config.num_wires)
        .flat_map(|w| (0..common_data.degree())
            .map(move |r| witness.get_wire(r, w)))
        .collect();

    // 4. Serialize k_is, subgroup
    let k_is = &common_data.k_is;
    let subgroup = &prover_data.subgroup;

    // 5. Get precomputed constants_sigmas_commitment GPU data
    // (This requires the PolynomialBatch to have been built with GPU path)

    // 6. Allocate proof output buffer
    let mut proof_buf = vec![0u8; MAX_PROOF_SIZE];
    let mut proof_size: usize = 0;

    // 7. Call FFI
    let err = unsafe {
        gpu_prove(
            /* pointers */,
            &config,
            gates.as_ptr(),
            gates.len() as u32,
            /* ... */
            proof_buf.as_mut_ptr() as *mut c_void,
            &mut proof_size,
            gpu_id as u64,
        )
    };

    if err.code != 0 {
        return Err(format!("GPU prover failed: {:?}", err));
    }

    // 8. Deserialize proof from proof_buf
    let proof = deserialize_proof(&proof_buf[..proof_size])?;
    Ok(proof)
}
```

### 3. Integration in plonky2 prover.rs

In `plonky2/plonky2/src/plonk/prover.rs`, add a GPU dispatch:

```rust
#[cfg(feature = "cuda")]
pub fn prove_with_partition_witness<F, C, D>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where ...
{
    let witness = partition_witness.full_witness();
    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);

    // Try GPU prover, fall back to CPU on failure
    match zeknox::gpu_prove_safe(prover_data, common_data, &witness, &public_inputs, 0) {
        Ok(proof) => Ok(proof),
        Err(e) => {
            eprintln!("GPU prover failed ({}), falling back to CPU", e);
            prove_with_partition_witness_cpu(prover_data, common_data, partition_witness, timing)
        }
    }
}
```

### 4. Gate Type Mapping

Create a mapping from Plonky2 gate trait objects to the C enum:

```rust
fn gate_type_id(gate: &GateRef<F, D>) -> u32 {
    match gate.0.id().as_str() {
        "ArithmeticGate" => 0,
        "ArithmeticExtensionGate" => 1,
        "ConstantGate" => 2,
        "PublicInputGate" => 3,
        "PoseidonGate" => 4,
        "BaseSumGate" => 5,
        "RandomAccessGate" => 6,
        "NoopGate" => 7,
        _ => panic!("Unsupported gate type: {}", gate.0.id()),
    }
}
```

### 5. Proof Serialization Format

Define a binary format for the proof output that the Rust side can deserialize into `ProofWithPublicInputs`:
- Merkle caps (wires, zs_partial_products, quotient)
- Opening set values (extension field elements)
- FRI proof (commit phase caps, query round proofs, final poly, PoW witness)

Alternatively, pass structured data through the FFI boundary using additional output buffers.

## Dependencies

- Task 08 (Prover Orchestrator) -- the C function being wrapped
- Task 09 (CMake Integration) -- the library must be built

## Key Considerations

- The `constants_sigmas_commitment` is built once at circuit setup time and reused across multiple proofs. It should be precomputed and kept on GPU.
- Proof serialization/deserialization must be exact to ensure the verifier accepts the proof.
- The GPU prover output must be semantically identical to the CPU prover (same proof, same hash values).

## Testing

- Run end-to-end test: build a Plonky2 circuit, prove with GPU prover, verify with CPU verifier
- Compare proof bytes between CPU and GPU provers (they should match if using the same randomness, but may differ due to blinding -- verify semantically)
- Benchmark full prover pipeline: Rust overhead + GPU kernel time vs pure CPU time
