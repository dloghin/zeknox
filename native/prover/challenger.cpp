// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "prover/challenger.hpp"

#include <cassert>

#include "poseidon/poseidon_permutation.hpp"

typedef cpp_gl64_t GoldilocksField;

struct ChallengerImpl {
    PoseidonPermutation perm;
    std::vector<gl64_t> input_buffer;
    std::vector<gl64_t> output_buffer;

    void duplex()
    {
        const size_t n = input_buffer.size();
        assert(n <= CHALLENGER_SPONGE_RATE);

        perm.set_from_slice(input_buffer.data(), n, 0);
        input_buffer.clear();

        perm.permute();

        u64 out[CHALLENGER_SPONGE_WIDTH];
        perm.get_state_as_canonical_u64(out);
        output_buffer.clear();
        for (size_t i = 0; i < CHALLENGER_SPONGE_RATE; ++i) {
            output_buffer.push_back(gl64_t(out[i]));
        }
    }

    void observe_one(gl64_t elt)
    {
        output_buffer.clear();
        input_buffer.push_back(elt);
        if (input_buffer.size() == CHALLENGER_SPONGE_RATE) {
            duplex();
        }
    }
};

Challenger::Challenger() : impl(std::make_unique<ChallengerImpl>()) {}

Challenger::~Challenger() = default;

Challenger::Challenger(Challenger &&) noexcept = default;

Challenger &Challenger::operator=(Challenger &&) noexcept = default;

void Challenger::observe_hash(const gl64_t hash[NUM_HASH_OUT_ELTS]) { observe_elements(hash, NUM_HASH_OUT_ELTS); }

void Challenger::observe_cap(const gl64_t *cap, size_t cap_len) { observe_elements(cap, cap_len); }

void Challenger::observe_elements(const gl64_t *elts, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        impl->observe_one(elts[i]);
    }
}

void Challenger::observe_extension_elements(const gl64_t *elts, size_t count) { observe_elements(elts, count); }

void Challenger::observe_openings(const gl64_t *openings, size_t count) { observe_elements(openings, count); }

gl64_t Challenger::get_challenge()
{
    if (!impl->input_buffer.empty() || impl->output_buffer.empty()) {
        impl->duplex();
    }
    assert(!impl->output_buffer.empty());
    gl64_t ret = impl->output_buffer.back();
    impl->output_buffer.pop_back();
    return ret;
}

std::vector<gl64_t> Challenger::get_n_challenges(size_t n)
{
    std::vector<gl64_t> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(get_challenge());
    }
    return out;
}

void Challenger::get_extension_challenge(gl64_t out[2])
{
    out[0] = get_challenge();
    out[1] = get_challenge();
}
