# Task 05: Quotient Polynomial Evaluation CUDA Kernel

## Objective

Implement the CUDA kernel for `compute_quotient_polys` (prover.rs lines 600-744), the single heaviest computation in the prover. This evaluates the vanishing polynomial at all points in the quotient domain and produces the quotient polynomial.

## Background

The quotient polynomial `Q(x)` is defined as:
```
Q(x) = VanishingPoly(x) / Z_H(x)
```
where `VanishingPoly(x)` is a random linear combination of:
1. Gate constraints (evaluated per gate type using selectors)
2. Permutation argument terms: `L_0(x) * (Z(x) - 1)` and partial product checks
3. All combined with alpha powers

The evaluation domain has size `degree * quotient_degree_factor` (typically `degree * 8`).

## Source Reference

Key function: `compute_quotient_polys` in `prover.rs` (lines 600-744)
Key sub-function: `eval_vanishing_poly_base_batch` in `vanishing_poly.rs` (lines 118-222)

## Deliverables

### Files
- `native/prover/quotient_poly.cuh` -- kernel declarations and types
- `native/prover/quotient_poly.cu` -- kernel implementations
- `native/prover/gate_constraints.cuh` -- gate constraint evaluation kernels

### Architecture: Two-Phase Approach

#### Phase 1: Gate Constraint Evaluation

For each gate type, a specialized CUDA kernel evaluates that gate's constraints at all LDE domain points.

```cpp
// Generic interface for a gate constraint evaluator
struct GateDescriptor {
    uint32_t gate_type;          // enum identifying the gate
    uint32_t selector_index;     // which selector polynomial to use
    uint32_t group_start;        // selector group range
    uint32_t group_end;
    uint32_t num_selectors;
};

// Kernel: evaluate constraints for ArithmeticGate at all points
__global__ void eval_arithmetic_gate_constraints(
    const gl64_t* local_constants,   // constants LDE values
    const gl64_t* local_wires,       // wires LDE values
    size_t num_points,               // LDE domain size
    size_t num_constants,            // constants per point
    size_t num_wires,                // wires per point
    size_t step,                     // LDE step factor
    const GateDescriptor* gate_desc,
    gl64_t* constraint_accumulator,  // [num_gate_constraints x num_points]
    size_t num_gate_constraints
);
```

**Gate types to implement** (most common in Plonky2):
- `ArithmeticGate` -- `a * b * const_0 + c * const_1`
- `ArithmeticExtensionGate` -- same in extension field
- `ConstantGate` -- wire = constant
- `PublicInputGate` -- public input enforcement
- `PoseidonGate` -- Poseidon permutation constraint
- `BaseSumGate` -- base decomposition constraint
- `RandomAccessGate` -- memory access
- `NoopGate` -- no constraints (skip)

Each gate kernel reads the appropriate wire and constant values from the LDE arrays and writes to the shared constraint accumulator.

#### Phase 2: Vanishing Polynomial Evaluation

```cpp
__global__ void eval_vanishing_poly_kernel(
    // LDE data from all polynomial batches (GPU pointers)
    const gl64_t* constants_sigmas_lde,   // from prover_data
    const gl64_t* wires_lde,              // from wires_commitment
    const gl64_t* zs_partial_products_lde, // from zs commitment
    // Sizes
    size_t lde_size,                // = degree << quotient_degree_bits
    size_t degree,
    size_t step,                    // = 1 << (rate_bits - quotient_degree_bits)
    size_t next_step,               // = 1 << quotient_degree_bits
    // Ranges (offsets into leaf data)
    size_t constants_start, size_t constants_end,
    size_t sigmas_start, size_t sigmas_end,
    size_t zs_start, size_t zs_end,
    size_t partial_products_start, size_t partial_products_end,
    // Challenge values
    const gl64_t* betas,            // [num_challenges]
    const gl64_t* gammas,           // [num_challenges]
    const gl64_t* alphas,           // [num_challenges]
    const gl64_t* k_is,             // [num_routed_wires]
    size_t num_challenges,
    size_t num_routed_wires,
    size_t num_partial_products,
    size_t quotient_degree_factor,
    // Gate constraint results (from Phase 1)
    const gl64_t* gate_constraint_values,  // [num_gate_constraints x lde_size]
    size_t num_gate_constraints,
    // Z_H inverse values (precomputed)
    const gl64_t* z_h_inv,          // [lde_size]
    // Output
    gl64_t* quotient_values         // [num_challenges x lde_size]
);
```

Each thread handles one LDE point:

1. **Read LDE values**: Access `constants_sigmas_lde`, `wires_lde`, `zs_partial_products_lde` at the appropriate bit-reversed index (matching `get_lde_values(i, step)`)

2. **Compute L_0(x) * (Z(x) - 1)** for each challenge:
   - `L_0(x) = z_h_on_coset.eval_l_0(i, shifted_x)`
   - `vanishing_z_1 = L_0 * (Z(x) - 1)`

3. **Compute partial product checks** for each challenge:
   - Read `numerator_values[j] = wire + beta * k_is[j] * shifted_x + gamma`
   - Read `denominator_values[j] = wire + beta * s_sigma[j] + gamma`
   - Call `check_partial_products` with the current partial products, `z_x`, `z_gx`

4. **Combine constraint terms** with alpha powers:
   - `result = reduce_with_powers_multi(all_terms, alphas)`

5. **Divide by Z_H**: `quotient_values[i] = result * z_h_inv[i]`

### Z_H on Coset (precomputed)

```cpp
// Precompute Z_H inverse values on the quotient domain
__global__ void precompute_z_h_inverse_kernel(
    size_t degree_bits,
    size_t quotient_degree_bits,
    gl64_t* z_h_inv,               // [lde_size]
    size_t lde_size
);
```

`Z_H(x) = x^n - 1` on the coset. The inverse is precomputed once and reused.

### Final Step: Coset IFFT

After computing quotient values on the coset domain, perform:
1. Transpose the quotient values (from `[num_challenges x lde_size]` to per-polynomial layout)
2. Coset IFFT to get coefficient form: `values.coset_ifft(F::coset_shift())`
3. This reuses `compute_batched_ntt` with appropriate coset parameters

## Memory Access Pattern

The LDE data is stored in Merkle tree leaf layout (transposed, bit-reversed). For the quotient poly evaluation:
- `get_lde_values(i, step)` accesses `leaves[bit_reverse(i * step) * leaf_size]`
- This is a strided, non-sequential access pattern
- Consider reorganizing data for this kernel or using texture memory/L2 cache hints

## Performance Considerations

- LDE domain size can be `degree * 8` = up to 2^23 points. Each point evaluation is independent.
- Gate constraint evaluation: the Plonky2 `eval_filtered_base_batch` pattern uses batch evaluation. On GPU, each thread evaluates all gates at one point.
- Memory bound: reading LDE values from 3 different polynomial batches for each point.
- Consider fusing Phase 1 and Phase 2 if gate constraints can be evaluated inline.

## Data Dependencies

- **Input**: All three PolynomialBatchGPU instances (constants_sigmas, wires, zs_partial_products) with LDE data on GPU, challenge scalars from the Challenger, gate descriptors, k_is
- **Output**: Quotient polynomial coefficients on GPU, ready for `PolynomialBatchGPU::from_coeffs`

## Testing

- Compare quotient polynomial output against Plonky2's CPU `compute_quotient_polys` for a test circuit
- Verify the quotient polynomial has the expected degree: `trim_to_len(quotient_degree)` should succeed (no remainder)
- Test with different gate types to ensure each gate kernel is correct
