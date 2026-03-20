# Task 01: PolynomialBatchGPU C++ Wrapper

## Objective

Create a `PolynomialBatchGPU` C++ class that wraps the existing GPU primitives (LDE, transpose, Merkle tree) into a unified `from_values` / `from_coeffs` pipeline, mirroring the Rust `PolynomialBatch` struct.

## Background

In `plonky2/plonky2/src/fri/oracle.rs`, the `PolynomialBatch` struct holds:
- `polynomials: Vec<PolynomialCoeffs<F>>` -- coefficient-form polynomials
- `merkle_tree: MerkleTree<F, C::Hasher>` -- Merkle tree over LDE values
- `degree_log`, `rate_bits`, `blinding`

The existing Rust GPU path (`from_coeffs_gpu`, lines 264-387) already chains:
1. `compute_batched_lde` / `compute_batched_lde_multi_gpu` (from `native/ntt/ntt.cu`)
2. `compute_transpose_rev` (from `native/ntt/ntt.cu`)
3. `fill_digests_buf_linear_gpu_with_gpu_ptr` (from `native/merkle/merkle.cu`)

## Deliverables

### File: `native/prover/polynomial_batch.cuh`

```cpp
class PolynomialBatchGPU {
public:
    // GPU pointers to LDE data (transposed, bit-reversed -- Merkle leaf layout)
    gl64_t* lde_data_gpu;          // [num_leaves x leaf_size]
    size_t num_leaves;             // = 1 << (degree_log + rate_bits)
    size_t leaf_size;              // = num_polynomials + salt_size

    // Merkle tree outputs (GPU)
    gl64_t* digests_gpu;
    gl64_t* cap_gpu;
    size_t num_digests;
    size_t cap_len;

    // Polynomial coefficients (GPU)
    gl64_t* coeffs_gpu;            // [num_polynomials x degree]
    size_t num_polynomials;

    // Config
    size_t degree_log;
    size_t rate_bits;
    bool blinding;
    size_t cap_height;

    // Build from evaluation values (IFFT -> from_coeffs)
    static PolynomialBatchGPU from_values(
        gl64_t* values_gpu,        // [num_polys x degree], evaluations on subgroup
        size_t num_polys,
        size_t degree_log,
        size_t rate_bits,
        bool blinding,
        size_t cap_height,
        size_t gpu_id
    );

    // Build from coefficient form (LDE -> transpose -> Merkle)
    static PolynomialBatchGPU from_coeffs(
        gl64_t* coeffs_gpu,        // [num_polys x degree], coefficients
        size_t num_polys,
        size_t degree_log,
        size_t rate_bits,
        bool blinding,
        size_t cap_height,
        size_t gpu_id
    );

    // Access LDE values at a given index (for quotient poly evaluation)
    // Returns pointer to leaf at bit_reverse(index * step)
    const gl64_t* get_lde_values(size_t index, size_t step) const;

    ~PolynomialBatchGPU();  // frees GPU memory
};
```

## Implementation Details

### `from_values`
1. Call `compute_batched_ntt` with `NTT_Direction::inverse` to get coefficients (IFFT)
2. Delegate to `from_coeffs`

### `from_coeffs`
1. Configure `NTTConfig`:
   - `batches = num_polys`
   - `extension_rate_bits = rate_bits`
   - `are_inputs_on_device = true`
   - `are_outputs_on_device = true`
   - `with_coset = true`
   - `salt_size = blinding ? 4 : 0`
2. Call `compute_batched_lde` (or multi-GPU variant)
3. Configure `NTT_TransposeConfig`:
   - `batches = num_polys + salt_size`
   - `are_inputs_on_device = true`
   - `are_outputs_on_device = true`
4. Call `compute_transpose_rev`
5. Call `fill_digests_buf_linear_gpu_with_gpu_ptr` with the transposed data

### `get_lde_values`
- Compute `actual_index = reverse_bits(index * step, degree_log + rate_bits)`
- Return `&lde_data_gpu[actual_index * leaf_size]`
- Strip salt columns if `blinding` is true (return only first `num_polynomials` elements)

## Existing Code to Reuse

- `native/ntt/ntt.cu`: `batch_ntt`, `batch_lde`, `batch_lde_multi_gpu`
- `native/ntt/ntt.h`: `NTT_Config`, `NTT_Direction`, `NTT_TransposeConfig`
- `native/merkle/merkle.h`: `fill_digests_buf_linear_gpu_with_gpu_ptr`
- `native/ff/gl64_t.cuh`: Goldilocks field type
- `wrappers/rust/src/lib.rs` (lines ~100-180): Reference for how the Rust wrapper calls these

## Testing

- Create a test that builds a `PolynomialBatchGPU` from known polynomial values, copies the Merkle cap back to host, and compares against Plonky2's CPU `MerkleTree::new_from_2d` output.
- Verify LDE values match the CPU path by comparing `get_lde_values` output at random indices.
