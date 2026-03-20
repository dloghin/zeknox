# Task 07: Opening Set Construction

## Objective

Implement polynomial evaluation at extension field points (`zeta` and `g * zeta`) for the opening set construction, which produces the polynomial openings included in the Plonky2 proof.

## Background

After the quotient polynomial is committed, the verifier challenges with a point `zeta` in the extension field. The prover evaluates all committed polynomials at `zeta` and `g * zeta` (where `g` is the generator of the evaluation domain).

From `plonky2/plonky2/src/plonk/proof.rs`, the `OpeningSet` contains:
- `constants`: evaluations of constant polynomials at `zeta`
- `plonk_sigmas`: evaluations of sigma polynomials at `zeta`
- `wires`: evaluations of wire polynomials at `zeta`
- `plonk_zs`: evaluations of Z polynomials at `zeta`
- `plonk_zs_next`: evaluations of Z polynomials at `g * zeta`
- `partial_products`: evaluations of partial product polynomials at `zeta`
- `quotient_polys`: evaluations of quotient polynomial chunks at `zeta`

## Source Reference

```rust
// plonk/proof.rs -- OpeningSet::new
pub fn new(
    zeta: F::Extension,
    g: F::Extension,
    constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    zs_partial_products_commitment: &PolynomialBatch<F, C, D>,
    quotient_polys_commitment: &PolynomialBatch<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Self {
    // For each polynomial, evaluate at zeta using Horner's method on coefficients
    let eval_commitment = |z: F::Extension, c: &PolynomialBatch<F, C, D>| -> Vec<F::Extension> {
        c.polynomials.iter()
            .map(|p| p.eval(z))
            .collect()
    };
    // ...
}
```

## Deliverables

### Files
- `native/prover/opening_set.cuh` -- declarations
- `native/prover/opening_set.cu` -- implementation

### CUDA Kernel: Polynomial Evaluation at Extension Field Point

```cpp
// Evaluate multiple polynomials at a single extension field point
// Each polynomial is in coefficient form: p(x) = sum_i coeffs[i] * x^i
// Evaluation uses Horner's method: p(x) = coeffs[n-1] + x*(coeffs[n-2] + x*(...))
__global__ void eval_polynomials_at_point(
    const gl64_t* coeffs_gpu,       // [num_polys x degree], interleaved or batched
    size_t num_polys,
    size_t degree,
    const gl64_t zeta[2],           // extension field point (D=2)
    gl64_t* results                 // [num_polys * 2], extension field results
);
```

Each thread handles one polynomial:
1. Load `zeta` (extension field point)
2. Horner evaluation: `result = 0; for i in (0..degree).rev(): result = result * zeta + coeffs[poly_idx * degree + i]`
3. Write the extension field result (2 Goldilocks elements)

Note: The coefficients are base field elements, but the evaluation point and result are extension field elements. So the multiplication is `gl64_ext2_t * gl64_t` (scalar * extension) which simplifies to component-wise base field multiply.

### Host Function: Construct Opening Set

```cpp
struct OpeningSet {
    std::vector<gl64_ext2_t> constants;
    std::vector<gl64_ext2_t> plonk_sigmas;
    std::vector<gl64_ext2_t> wires;
    std::vector<gl64_ext2_t> plonk_zs;
    std::vector<gl64_ext2_t> plonk_zs_next;
    std::vector<gl64_ext2_t> partial_products;
    std::vector<gl64_ext2_t> quotient_polys;
};

OpeningSet construct_opening_set(
    gl64_ext2_t zeta,
    gl64_ext2_t g,                    // domain generator
    const PolynomialBatchGPU& constants_sigmas,
    const PolynomialBatchGPU& wires,
    const PolynomialBatchGPU& zs_partial_products,
    const PolynomialBatchGPU& quotient_polys,
    // Range information (which polynomial indices correspond to which oracle)
    size_t constants_start, size_t constants_end,
    size_t sigmas_start, size_t sigmas_end,
    size_t zs_start, size_t zs_end,
    size_t partial_products_start, size_t partial_products_end,
    size_t gpu_id
);
```

Steps:
1. Evaluate all polynomials in `constants_sigmas` at `zeta`:
   - Split results into `constants` and `plonk_sigmas` using range info
2. Evaluate all polynomials in `wires` at `zeta`
3. Evaluate all polynomials in `zs_partial_products` at `zeta` and at `g * zeta`:
   - At `zeta`: split into `plonk_zs` and `partial_products`
   - At `g * zeta`: extract `plonk_zs_next`
4. Evaluate all polynomials in `quotient_polys` at `zeta`
5. Transfer results from GPU to host

## Performance Considerations

- Number of polynomials is moderate (typically 100-500). Each evaluation is O(degree) multiplications.
- Degree can be 2^14 to 2^20, so each Horner evaluation is significant.
- All evaluations at the same point are independent -- parallelize across polynomials.
- For very large degree, consider using shared memory for partial accumulation within a polynomial.

## Dependencies

- Task 01 (PolynomialBatchGPU) -- for accessing polynomial coefficients on GPU
- Task 03 (gl64_ext2_t) -- for extension field arithmetic

## Testing

- Evaluate known polynomials at known points and verify against hand-computed results
- Cross-validate with Plonky2's `PolynomialCoeffs::eval` on the same polynomials and points
- Verify that all opening values are consistent with the committed polynomial batches
