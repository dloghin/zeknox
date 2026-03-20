# Task 06: FRI Commit-Phase Folding CUDA Kernel

## Objective

Implement CUDA kernels for the FRI (Fast Reed-Solomon Interactive Oracle Proof) commit phase, which performs polynomial folding and Merkle tree construction at each reduction round.

## Background

The FRI commit phase (from `plonky2/plonky2/src/fri/prover.rs`, `fri_committed_trees` function, lines 69-113) iterates over `reduction_arity_bits` and at each round:

1. Bit-reverse the polynomial values
2. Chunk values into groups of `arity = 2^arity_bits`
3. Build a Merkle tree over the chunked (flattened) values
4. Sample a challenge `beta` from the challenger
5. Fold the polynomial: `P(x) = sum_{i<arity} beta^i * P_i(x^arity)` where `P_i` are the chunks of the coefficient representation
6. Evaluate the folded polynomial on the reduced coset domain

## Source Reference

```rust
// fri/prover.rs lines 69-113
fn fri_committed_trees<F, C, D>(...) -> FriCommitedTrees<F, C, D> {
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        reverse_index_bits_in_place(&mut values.values);
        let chunked_values = values.values
            .par_chunks(arity)
            .map(|chunk| flatten(chunk))
            .collect();
        let tree = MerkleTree::new_from_2d(chunked_values, cap_height);

        challenger.observe_cap(&tree.cap);
        let beta = challenger.get_extension_challenge();

        // Polynomial folding in coefficient form
        coeffs = PolynomialCoeffs::new(
            coeffs.coeffs
                .par_chunks_exact(arity)
                .map(|chunk| reduce_with_powers(chunk, beta))
                .collect()
        );
        shift = shift.exp_u64(arity as u64);
        values = coeffs.coset_fft(shift.into());
    }
    // ...
}
```

## Deliverables

### Files
- `native/prover/fri_fold.cuh` -- kernel declarations
- `native/prover/fri_fold.cu` -- kernel implementations

### Kernel 1: Bit-Reverse and Chunk for Merkle Tree

```cpp
// Bit-reverse values in-place and prepare chunked leaves for Merkle tree
__global__ void fri_prepare_merkle_leaves(
    const gl64_t* values,           // [n] extension field values (2 * n gl64_t elements)
    gl64_t* leaves_out,             // [n/arity * arity * D] flattened leaves
    size_t n,                       // number of extension field elements
    size_t arity_bits,              // log2(arity)
    size_t ext_degree               // D = 2 for quadratic extension
);
```

Each thread handles one chunk:
1. For elements in the chunk, bit-reverse their indices
2. Flatten the extension field elements (each has `D=2` base field components)
3. Write to `leaves_out` in Merkle-leaf-ready layout

After this, call `fill_digests_buf_linear_gpu_with_gpu_ptr` (existing) to build the Merkle tree.

### Kernel 2: Polynomial Folding (Coefficient-Domain)

```cpp
// Fold polynomial coefficients with beta challenge
// coeffs_in has n coefficients, output has n/arity coefficients
__global__ void fri_fold_coefficients(
    const gl64_t* coeffs_in,        // [n * D] extension field coefficients
    gl64_t* coeffs_out,             // [n/arity * D] folded coefficients
    const gl64_t beta[2],           // extension field challenge (D=2)
    size_t n,                       // input coefficient count
    size_t arity_bits,              // log2(arity)
    size_t ext_degree               // D = 2
);
```

Each thread handles one output coefficient:
1. Read `arity` input coefficients: `coeffs_in[tid * arity .. (tid+1) * arity]`
2. Compute `result = reduce_with_powers(chunk, beta)`:
   ```
   result = chunk[arity-1]
   for i in (0..arity-1).rev():
       result = result * beta + chunk[i]
   ```
3. Write `result` to `coeffs_out[tid]`

All arithmetic is in the extension field (`gl64_ext2_t`).

### Kernel 3: Coset FFT for Folded Polynomial

After folding, evaluate the folded polynomial on the reduced coset:
- `values = coeffs.coset_fft(shift.into())`
- `shift = shift.exp_u64(arity)`

This is an NTT on extension field elements. Options:
1. **Reuse existing NTT**: The existing `compute_batched_ntt` works on `gl64_t`. For extension field, treat each extension element as 2 independent NTTs (component-wise NTT is valid for the "evaluate polynomial" operation since the extension is defined over the base field's subgroup).
2. **Extension field NTT**: Write a new NTT kernel for `gl64_ext2_t`. More complex but avoids data reshuffling.

**Recommended**: Option 1 -- decompose each extension field coefficient into real and imaginary parts, run two base-field NTTs, then recombine. The coset shift is a base field element, so component-wise coset FFT is correct.

## FRI Commit Phase Orchestrator

```cpp
struct FriCommitPhaseResult {
    std::vector<MerkleTreeGPU> trees;    // one per round
    gl64_ext2_t* final_coeffs;           // GPU pointer to final polynomial
    size_t final_degree;
};

FriCommitPhaseResult fri_commit_phase(
    gl64_ext2_t* lde_coeffs_gpu,          // initial polynomial coefficients
    gl64_ext2_t* lde_values_gpu,          // initial polynomial values
    size_t initial_degree,
    const std::vector<size_t>& reduction_arity_bits,
    size_t rate_bits,
    size_t cap_height,
    Challenger& challenger,               // for observe_cap / get beta
    size_t gpu_id
);
```

Loop:
1. `fri_prepare_merkle_leaves` -> build Merkle tree (reuse existing) -> challenger observes cap
2. `beta = challenger.get_extension_challenge()`
3. `fri_fold_coefficients` with beta
4. Coset FFT on folded coefficients
5. Repeat with reduced size

## Data Flow

```
Input: final_poly coefficients + values from prove_openings

Round 1:
  values (size N) -> bit-reverse -> chunk -> Merkle tree -> cap -> challenger -> beta
  coeffs (size N) -> fold with beta -> coeffs (size N/arity)
  coeffs -> coset_fft(shift^arity) -> values (size N/arity)

Round 2:
  values (size N/arity) -> ... -> Merkle tree -> cap -> beta
  coeffs -> fold -> coeffs (size N/arity^2)
  ...

Final:
  coeffs (small) -> truncate to degree = n >> rate_bits -> observe in challenger
```

## Performance Considerations

- FRI typically has 3-5 rounds with arities 2-4
- First round is the largest (full LDE size). Subsequent rounds halve or quarter.
- Merkle tree building dominates cost at each round (reuses existing efficient GPU kernel)
- The folding kernel is lightweight compared to Merkle tree hashing
- Extension field NTT is 2x the cost of base field NTT

## Dependencies

- Task 02 (Challenger) -- needed for observe_cap / get_extension_challenge
- Task 03 (gl64_ext2_t) -- extension field arithmetic for folding
- Task 01 (PolynomialBatchGPU) -- Merkle tree building

## Testing

- Compare FRI commit phase output (Merkle caps, final coefficients) against Plonky2's CPU `fri_committed_trees`
- Verify that folded polynomial evaluations are consistent: `P(x) = sum beta^i P_i(x^arity)` at random points
- Test with typical FRI parameters: `reduction_arity_bits = [3, 3, 3]`, `rate_bits = 3`
