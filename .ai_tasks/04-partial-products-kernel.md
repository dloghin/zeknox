# Task 04: Partial Products + Z Polynomial CUDA Kernel

## Objective

Implement the CUDA kernel for `wires_permutation_partial_products_and_zs` (prover.rs lines 383-440), which computes the permutation argument's partial product polynomials and the Z polynomial used in the Plonky2 copy constraint check.

## Background

The permutation argument proves that wires are correctly connected across gates. For each challenge `(beta, gamma)`:

For each row `i`:
- Numerator: `prod_j(wire[i][j] + beta * k_is[j] * subgroup[i] + gamma)` over `j in 0..num_routed_wires`
- Denominator: `prod_j(wire[i][j] + beta * sigmas[i][j] + gamma)` over `j in 0..num_routed_wires`
- Quotient: `numerator / denominator`

The quotient is split into chunks (of size `quotient_degree_factor`), and a running product Z is maintained:
- `Z(g^0) = 1`
- `Z(g^{i+1}) = Z(g^i) * product_of_quotients_at_row_i`

## Source Reference

```rust
// prover.rs lines 383-440
fn wires_permutation_partial_products_and_zs(...) -> Vec<PolynomialValues<F>> {
    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            // numerators[j] = wire_value + beta * k_is[j] * x + gamma
            // denominators[j] = wire_value + beta * sigmas[i][j] + gamma
            // quotient_values[j] = numerators[j] * denominator_inv[j]
            // return quotient_chunk_products(&quotient_values, degree)
        })
        .collect();

    // Sequential prefix product for Z
    let mut z_x = F::ONE;
    for quotient_chunk_products in all_quotient_chunk_products {
        let partial_products_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunk_products);
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }
    // transpose and wrap into PolynomialValues
}
```

## Deliverables

### Files
- `native/prover/partial_products.cuh` -- kernel declarations
- `native/prover/partial_products.cu` -- kernel implementations

### Kernel Design

#### Phase A: Per-Row Quotient Computation (massively parallel)

```
__global__ void compute_quotient_chunk_products_kernel(
    const gl64_t* wire_values,       // [degree x num_wires]
    const gl64_t* sigmas,            // [degree x num_routed_wires]
    const gl64_t* subgroup,          // [degree]
    const gl64_t* k_is,              // [num_routed_wires]
    gl64_t beta,
    gl64_t gamma,
    size_t degree,
    size_t num_wires,
    size_t num_routed_wires,
    size_t quotient_degree_factor,   // chunk size
    gl64_t* chunk_products_out       // [degree x num_chunks]
);
```

Each thread handles one row:
1. Compute `numerator[j] = wire_values[i * num_wires + j] + beta * k_is[j] * subgroup[i] + gamma` for `j in 0..num_routed_wires`
2. Compute `denominator[j] = wire_values[i * num_wires + j] + beta * sigmas[i * num_routed_wires + j] + gamma`
3. Batch invert denominators (Montgomery's trick within each thread using shared memory, or per-element inverse since `num_routed_wires` is typically small ~80)
4. Compute `quotient[j] = numerator[j] * denominator_inv[j]`
5. Compute chunk products: split quotients into chunks of `quotient_degree_factor`, multiply within each chunk
6. Write `chunk_products_out[i * num_chunks + c]` for each chunk `c`

#### Phase B: Prefix Product Scan for Z Polynomial

```
__global__ void compute_z_prefix_product_kernel(
    const gl64_t* chunk_products,    // [degree x num_chunks]
    size_t degree,
    size_t num_chunks,               // = num_partial_products + 1
    gl64_t* partial_products_out,    // [degree x (num_partial_products + 1)]
    gl64_t* z_poly_out               // [degree]  (the Z polynomial values)
);
```

The prefix product has a sequential dependency across rows. Options:
1. **CPU fallback**: Copy chunk products to host, compute prefix product, copy back. Simple but involves transfer.
2. **GPU Blelloch scan**: Parallelize using a work-efficient prefix scan in Goldilocks arithmetic. This is O(n) work, O(log n) depth.
3. **Hybrid**: Compute partial products within blocks on GPU, then combine block results on CPU.

**Recommended**: Start with option 1 (CPU prefix product) for correctness, then optimize to option 2.

The prefix product logic (from `partial_products_and_z_gx`):
```
z_x = 1
for i in 0..degree:
    partial_products_and_z_gx = [z_x * chunk_products[i][0],
                                  z_x * chunk_products[i][0] * chunk_products[i][1],
                                  ...,
                                  z_x * prod(chunk_products[i][0..num_chunks])]
    // The last element becomes z_{i+1}
    z_x = partial_products_and_z_gx[num_prods]
```

#### Output Layout

The output is `num_challenges * (num_partial_products + 1)` polynomials, each of length `degree`. These need to be in evaluation form (PolynomialValues).

Layout: For challenge `c`:
- `z_poly[c]`: polynomial values `[z_x_0, z_x_1, ..., z_x_{degree-1}]`
- `partial_product[c][p]`: polynomial values for partial product `p`

The Z polynomial goes first in the batch (per `zs_range` convention), followed by partial products.

## Data Dependencies

- **Input**: `wire_values` (from witness, already on GPU), `sigmas` (precomputed, need to upload), `subgroup` (can precompute on GPU), `k_is` (small, upload once), `beta`/`gamma` (from challenger, host scalars)
- **Output**: Evaluation-form polynomials on GPU, ready for `PolynomialBatchGPU::from_values`

## Performance Considerations

- `num_routed_wires` is typically 60-80 for standard Plonky2 circuits. This means each thread does ~80 multiplications + a batch inverse. Fits in registers.
- `degree` can be 2^14 to 2^20. Plenty of parallelism.
- The batch inverse within a single row can use Montgomery's trick: compute prefix products, one inversion, then suffix products. This is O(3n) multiplications + 1 inversion.
- Memory: `wire_values` and `sigmas` are the large arrays. Ensure coalesced access patterns.

## Testing

- Compare output against Plonky2's `all_wires_permutation_partial_products` for a known circuit
- Verify `Z(g^0) = 1` and `Z(g^n) = 1` (the permutation argument completeness check)
- Test with `num_challenges = 2` (Plonky2 default)
