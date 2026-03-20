# Task 08: Top-Level Prover Orchestrator

## Objective

Write the top-level `gpu_prove()` C function in `native/prover/prover.cu` that orchestrates all prover stages, keeping data on GPU between stages and minimizing host-device transfers.

## Background

This is the main entry point that chains together all the components built in Tasks 01-07 to replicate the full `prove_with_partition_witness` function from `prover.rs` (lines 123-353).

## Deliverables

### Files
- `native/prover/prover.h` -- C API header
- `native/prover/prover.cu` -- Implementation
- `native/prover/types.h` -- Data structures for C API

### C API

```cpp
// types.h
struct ProverConfig {
    uint32_t degree_bits;
    uint32_t num_wires;
    uint32_t num_routed_wires;
    uint32_t num_challenges;
    uint32_t num_partial_products;
    uint32_t quotient_degree_factor;
    uint32_t rate_bits;
    uint32_t cap_height;
    uint32_t num_gate_constraints;
    uint32_t num_constants;
    uint32_t num_public_inputs;
};

struct GateInfo {
    uint32_t gate_type;
    uint32_t selector_index;
    uint32_t group_start;
    uint32_t group_end;
    uint32_t num_selectors;
    uint32_t num_constraints;
};

// prover.h
EXTERN RustError gpu_prove(
    // Precomputed circuit data (GPU pointers)
    const void* constants_sigmas_coeffs_gpu,  // [num_const_sigma_polys x degree]
    const void* constants_sigmas_lde_gpu,     // precomputed LDE (transposed, bit-rev)
    const void* constants_sigmas_digests_gpu, // precomputed Merkle digests
    const void* constants_sigmas_cap_gpu,     // precomputed Merkle cap

    // Circuit metadata
    const void* circuit_digest,    // [NUM_HASH_OUT_ELTS] Goldilocks elements
    const void* public_inputs_hash,// [NUM_HASH_OUT_ELTS]
    const ProverConfig* config,
    const GateInfo* gates,
    uint32_t num_gates,
    const void* k_is,              // [num_routed_wires] coset shifts
    const void* subgroup,          // [degree] subgroup elements

    // Witness (host pointer, will be copied to GPU)
    const void* wire_values,       // [num_wires x degree] (column-major)

    // FRI parameters
    const uint32_t* reduction_arity_bits,
    uint32_t num_fri_rounds,
    const void* fft_root_table,    // optional precomputed roots

    // Output (host pointer, filled by the function)
    void* proof_output,
    size_t* proof_size,

    // GPU selection
    uint64_t gpu_id
);
```

### Orchestration Flow

```cpp
RustError gpu_prove(...) {
    // === STAGE 1: Transfer witness to GPU ===
    gl64_t* wire_values_gpu = cuda_malloc(num_wires * degree);
    cuda_memcpy_h2d(wire_values_gpu, wire_values, ...);

    // === STAGE 2: Wires Commitment ===
    // Convert wire columns to PolynomialValues format
    auto wires_commitment = PolynomialBatchGPU::from_values(
        wire_values_gpu, num_wires, degree_bits, rate_bits,
        blinding, cap_height, gpu_id);

    // === STAGE 3: Challenger setup ===
    Challenger challenger;
    challenger.observe_hash(circuit_digest);
    challenger.observe_hash(public_inputs_hash);
    challenger.observe_cap(wires_commitment.cap_gpu, wires_commitment.cap_len);

    auto betas = challenger.get_n_challenges(num_challenges);
    auto gammas = challenger.get_n_challenges(num_challenges);

    // === STAGE 4: Partial Products + Z ===
    gl64_t* partial_products_gpu;
    compute_all_partial_products(
        wire_values_gpu, sigmas_gpu, subgroup_gpu, k_is_gpu,
        betas, gammas, degree, num_wires, num_routed_wires,
        num_challenges, quotient_degree_factor,
        &partial_products_gpu);

    // === STAGE 5: Z's + Partial Products Commitment ===
    auto zs_pp_commitment = PolynomialBatchGPU::from_values(
        partial_products_gpu, num_zs_pp_polys, degree_bits,
        rate_bits, blinding, cap_height, gpu_id);

    challenger.observe_cap(zs_pp_commitment.cap_gpu, zs_pp_commitment.cap_len);
    auto alphas = challenger.get_n_challenges(num_challenges);

    // === STAGE 6: Quotient Polynomial ===
    gl64_t* quotient_coeffs_gpu;
    compute_quotient_polys(
        constants_sigmas_lde_gpu, wires_commitment, zs_pp_commitment,
        betas, gammas, alphas, k_is_gpu,
        gates, num_gates, config,
        &quotient_coeffs_gpu);

    // Split quotient into degree-n chunks
    // (quotient has degree quotient_degree, split into quotient_degree_factor chunks)

    // === STAGE 7: Quotient Commitment ===
    auto quotient_commitment = PolynomialBatchGPU::from_coeffs(
        quotient_chunks_gpu, num_chunks, degree_bits,
        rate_bits, blinding, cap_height, gpu_id);

    challenger.observe_cap(quotient_commitment.cap_gpu, quotient_commitment.cap_len);

    // === STAGE 8: Opening Point ===
    gl64_ext2_t zeta;
    challenger.get_extension_challenge(zeta);

    // Verify zeta^n != 1
    gl64_ext2_t zeta_pow_n = zeta.exp_power_of_2(degree_bits);
    if (zeta_pow_n == gl64_ext2_t::one()) return error("zeta in subgroup");

    gl64_ext2_t g = gl64_ext2_t::primitive_root_of_unity(degree_bits);

    // === STAGE 9: Opening Set ===
    auto openings = construct_opening_set(
        zeta, g,
        constants_sigmas_commitment, wires_commitment,
        zs_pp_commitment, quotient_commitment, config);

    challenger.observe_openings(openings);

    // === STAGE 10: FRI prove_openings ===
    // Compute composition polynomial and FRI proof
    gl64_ext2_t alpha;
    challenger.get_extension_challenge(alpha);

    // Build final_poly = sum_batch alpha^k * (F_i(X) - F_i(z_i)) / (X - z_i)
    // LDE of final_poly
    // FRI commit phase (Task 06)
    auto fri_result = fri_commit_phase(...);

    // FRI PoW (simple brute-force on CPU)
    uint64_t pow_witness = fri_proof_of_work(challenger, config);

    // FRI query phase (CPU, uses Merkle proofs)
    auto query_proofs = fri_query_rounds(
        initial_merkle_trees, fri_result.trees,
        challenger, lde_size, fri_params);

    // === STAGE 11: Assemble Proof ===
    // Serialize: wires_cap, zs_pp_cap, quotient_cap, openings, FRI proof
    // Copy Merkle caps from GPU to host
    // Write to proof_output buffer

    // === Cleanup ===
    // Free all GPU allocations

    return RustError{0};
}
```

### GPU Memory Lifecycle

```
wire_values_gpu          -- allocated in Stage 1, freed after Stage 6
wires_commitment.lde_gpu -- allocated in Stage 2, needed until Stage 10 (FRI queries)
partial_products_gpu     -- allocated in Stage 4, freed after Stage 5
zs_pp_commitment.lde_gpu -- allocated in Stage 5, needed until Stage 10
quotient_coeffs_gpu      -- allocated in Stage 6, freed after Stage 7
quotient_commitment.lde_gpu -- allocated in Stage 7, needed until Stage 10
fri trees                -- allocated per round in Stage 10, needed for queries

Peak GPU memory: all 4 PolynomialBatch LDE arrays + working buffers
```

### Error Handling

Use `RustError` from `native/utils/rusterror.h` for FFI-compatible error reporting. Wrap CUDA calls with error checking macros.

## Performance Targets

- The primary goal is to beat the CPU prover by offloading LDE, Merkle tree, and quotient polynomial evaluation to GPU
- Secondary: keep data on GPU to avoid redundant host-device transfers
- The existing GPU LDE + Merkle pipeline (from `from_coeffs_gpu`) already achieves significant speedup; this task extends that to the full prover

## Dependencies

- Task 01 (PolynomialBatchGPU)
- Task 02 (Challenger)
- Task 03 (gl64_ext2_t)
- Task 04 (Partial Products Kernel)
- Task 05 (Quotient Polynomial Kernel)
- Task 06 (FRI Fold Kernel)
- Task 07 (Opening Set)

## Testing

- Run the complete prover on a test circuit and compare the output proof against Plonky2's CPU prover
- Verify the proof using Plonky2's verifier (the proof format must be byte-identical or at least semantically equivalent)
- Benchmark against the CPU prover and the hybrid Rust+GPU prover (existing `from_coeffs_gpu` path)
