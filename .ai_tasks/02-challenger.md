# Task 02: Challenger (Poseidon Sponge Fiat-Shamir)

## Objective

Implement a `Challenger` C++ class that mirrors the Rust `Challenger<F, Hasher>` from `plonky2/plonky2/src/iop/challenger.rs`, using the existing Poseidon permutation from `native/poseidon/poseidon.hpp`.

## Background

The Plonky2 challenger is a Fiat-Shamir transcript based on a Poseidon sponge. It:
- Absorbs field elements, hashes, and Merkle caps
- Squeezes out challenges (base field and extension field)
- Maintains an internal sponge state

The prover uses it at multiple points:
1. Observe circuit digest + public inputs hash
2. Observe wires cap -> squeeze betas, gammas
3. Observe Z's + partial products cap -> squeeze alphas
4. Observe quotient cap -> squeeze zeta (extension field)
5. Observe openings -> squeeze alpha for FRI composition
6. FRI commit phase: observe caps, squeeze betas

## Deliverables

### File: `native/prover/challenger.hpp`

```cpp
#include <poseidon/poseidon.hpp>
#include <ff/gl64_t.cuh>

// Plonky2 Poseidon sponge parameters
static constexpr size_t SPONGE_RATE = 8;
static constexpr size_t SPONGE_CAPACITY = 4;
static constexpr size_t SPONGE_WIDTH = 12;  // RATE + CAPACITY
static constexpr size_t NUM_HASH_OUT_ELTS = 4;

class Challenger {
private:
    gl64_t sponge_state[SPONGE_WIDTH];
    std::vector<gl64_t> input_buffer;
    std::vector<gl64_t> output_buffer;

    void duplex();  // absorb input_buffer, squeeze to output_buffer

public:
    Challenger();

    // Absorb a hash (4 field elements)
    void observe_hash(const gl64_t hash[NUM_HASH_OUT_ELTS]);

    // Absorb a Merkle cap (vector of hashes)
    void observe_cap(const gl64_t* cap, size_t cap_len);

    // Absorb field elements
    void observe_elements(const gl64_t* elts, size_t count);

    // Absorb extension field elements
    void observe_extension_elements(const gl64_t* elts, size_t count);

    // Absorb openings (FRI openings structure)
    void observe_openings(const gl64_t* openings, size_t count);

    // Squeeze one challenge
    gl64_t get_challenge();

    // Squeeze n challenges
    std::vector<gl64_t> get_n_challenges(size_t n);

    // Squeeze one extension field challenge (D=2)
    void get_extension_challenge(gl64_t out[2]);
};
```

## Implementation Details

### `duplex()`
The core sponge operation:
1. For each element in `input_buffer` (up to `SPONGE_RATE`):
   - `sponge_state[i] = input_buffer[i]`
2. Apply the Poseidon permutation to `sponge_state` (12 elements)
3. Fill `output_buffer` from `sponge_state[0..SPONGE_RATE]`
4. Clear `input_buffer`

### `observe_hash` / `observe_cap`
- Push the field elements into `input_buffer`
- When `input_buffer` reaches `SPONGE_RATE`, call `duplex()`
- Note: hashes are `NUM_HASH_OUT_ELTS = 4` Goldilocks elements each

### `get_challenge()`
1. If `input_buffer` is not empty, call `duplex()`
2. If `output_buffer` is empty, call `duplex()` (with zero-padded input)
3. Pop and return from `output_buffer`

### `get_extension_challenge()`
- Call `get_challenge()` twice to get the real and imaginary parts of the quadratic extension element

### Poseidon Permutation

Use the existing `PoseidonPermutation` from `native/poseidon/poseidon.hpp`. The Plonky2 Poseidon uses:
- Width = 12 field elements
- Full rounds = 8, partial rounds = 22
- Round constants and MDS matrix from `native/poseidon/poseidon.hpp`

This runs entirely on the **host** (CPU) since it processes small amounts of data sequentially.

## Existing Code to Reuse

- `native/poseidon/poseidon.hpp`: Poseidon permutation constants and CPU implementation
- `native/poseidon/poseidon.cpp`: `cpu_poseidon_hash_one` or equivalent
- `plonky2/plonky2/src/iop/challenger.rs`: Reference Rust implementation

## Key Rust Reference

```rust
// From plonky2/src/iop/challenger.rs (conceptual)
fn duplex(&mut self) {
    for i in 0..self.input_buffer.len().min(SPONGE_RATE) {
        self.sponge_state.set(i, self.input_buffer[i]);
    }
    self.input_buffer.clear();
    self.sponge_state = H::permute(self.sponge_state);
    self.output_buffer = self.sponge_state.squeeze().to_vec();
}
```

## Testing

- Implement a test that feeds the same sequence of observations as a Plonky2 Rust test and verifies that the squeezed challenges match exactly.
- Edge cases: empty input, buffer overflow across multiple duplexes, extension challenges.
