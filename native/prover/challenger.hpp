// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_CHALLENGER_HPP__
#define __ZEKNOX_PROVER_CHALLENGER_HPP__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "merkle/hasher.hpp"

#include "ff/goldilocks.hpp"
#if !defined(USE_CUDA)
typedef cpp_gl64_t gl64_t;
#endif

// Plonky2 Poseidon sponge parameters (same as `poseidon.hpp` / Plonky2).
// Named with a prefix so we do not clash with `SPONGE_*` macros from CUDA headers.
static constexpr size_t CHALLENGER_SPONGE_RATE = 8;
static constexpr size_t CHALLENGER_SPONGE_CAPACITY = 4;
static constexpr size_t CHALLENGER_SPONGE_WIDTH = 12;
// NUM_HASH_OUT_ELTS: see `merkle/hasher.hpp`

struct ChallengerImpl;

/// Fiat–Shamir transcript using a Plonky2-style duplex sponge over Poseidon (host CPU).
/// Semantics mirror `plonky2::iop::challenger::Challenger` with `PoseidonHash`.
class Challenger {
private:
    std::unique_ptr<ChallengerImpl> impl;

public:
    Challenger();
    ~Challenger();

    Challenger(Challenger &&) noexcept;
    Challenger &operator=(Challenger &&) noexcept;

    Challenger(const Challenger &) = delete;
    Challenger &operator=(const Challenger &) = delete;

    void observe_hash(const gl64_t hash[NUM_HASH_OUT_ELTS]);

    void observe_cap(const gl64_t *cap, size_t cap_len);

    void observe_elements(const gl64_t *elts, size_t count);

    void observe_extension_elements(const gl64_t *elts, size_t count);

    void observe_openings(const gl64_t *openings, size_t count);

    gl64_t get_challenge();

    std::vector<gl64_t> get_n_challenges(size_t n);

    void get_extension_challenge(gl64_t out[2]);
};

#endif // __ZEKNOX_PROVER_CHALLENGER_HPP__
