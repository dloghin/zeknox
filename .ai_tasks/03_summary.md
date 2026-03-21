# Task 03 (gl64_ext2_t) — Implementation Summary

## Branch and delivery

Work is on branch **`feat/gl64-ext2`** (from **`dev`**), committed, and pushed to **`origin`**.

## What was implemented

### `native/prover/gl64_ext2.cuh`

- **Field:** Goldilocks quadratic extension `F_p[x]/(x^2 - 7)` with `p = 2^64 - 2^32 + 1`, matching Plonky2 `QuadraticExtension<GoldilocksField>`.
- **Representation:** `real + imag * x` with `x^2 = 7` (non-residue `W = 7`).
- **API:** Constructors, `zero()` / `one()`, `+`, `-`, `*` (extension × extension), scalar `gl64_t * gl64_ext2_t`, `neg()`, `inverse()`, `pow(uint64_t)`, `exp_power_of_2(size_t)`, `primitive_root_of_unity(size_t)`, `==` / `!=`.
- **`primitive_root_of_unity(n_log)`:** Uses Plonky2’s extension **power-of-two generator** imaginary limb `15659105665374529263` (real part `0`), with extension **two-adicity** `33` (`32 + 1`), i.e. `base.exp_power_of_2(33 - n_log)` as in `Field::primitive_root_of_unity` / `EXT_POWER_OF_TWO_GENERATOR` in `goldilocks_extensions.rs`.
- **Base inverse (`inv_base`):** Fermat exponentiation `a^(p-2) mod p` (binary square-and-multiply), shared by host `cpp_gl64_t` and device `gl64_t`.
- **Additive negation (`neg_gl64`):** Canonical `MOD - v` for `v ≠ 0`. Used for `neg()` and the imaginary part of `inverse()`. Relying on `cpp_gl64_t`’s `zero() - x` for negation was observed to be off by **`EPSILON` (`2^32 - 1`)** in some cases relative to true field negation; canonical negation avoids that.
- **Includes:** `ff/gl64_t.cuh` then `ff/goldilocks.hpp` (explicit base-field header plus extension helpers); when `USE_CUDA` is off, `typedef cpp_gl64_t gl64_t` is provided for the same pattern as elsewhere in the tree.
- **Host/device:** `GL64_EXT2_HD` / `GL64_EXT2_INLINE` map to `__host__ __device__` under CUDA/HIP.

### `native/tests/tests.cu`

New **`Gl64Ext2`** tests:

- **`multiply_matches_plonky2`** — `(1+2x)(3+4x) → 59 + 10x` (integer coefficients).
- **`inverse_times_self_is_one`** — `(3,0)` and `(12345, 67890)` multiplied by their inverses yield `(1,0)` (canonical limbs).
- **`primitive_root_of_unity_order`** — for `n_log = 1..8`, `root.pow(1 << n_log) == one()`.
- **`neg_and_sub`** — `a + neg(a) == 0`.
- **`scalar_mul`** — `3 * (2+3x) == (6+9x)`.

## Notes

- **`extension_field.cuh`** was not used (per task): Montgomery-style `Field<CONFIG>` does not match Goldilocks `gl64_t` on device; implementation is direct over `gl64_t`.
- Equality compares **`get_val()`** on both limbs to avoid host/device `operator==` warnings and implicit conversion quirks on `cpp_gl64_t`.

## Pull request

Remote PR URL (if applicable):  
<https://github.com/dloghin/zeknox/pull/new/feat/gl64-ext2>
