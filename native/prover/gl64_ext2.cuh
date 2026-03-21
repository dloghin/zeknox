// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_GL64_EXT2_CUH__
#define __ZEKNOX_PROVER_GL64_EXT2_CUH__

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ff/gl64_t.cuh"
#include "ff/goldilocks.hpp"

#if !defined(USE_CUDA)
typedef cpp_gl64_t gl64_t;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GL64_EXT2_HD __host__ __device__
#define GL64_EXT2_INLINE __host__ __device__ __forceinline__
#else
#define GL64_EXT2_HD
#define GL64_EXT2_INLINE inline
#endif

// Additive inverse in F_p (canonical representative in [0, p)).
static GL64_EXT2_INLINE gl64_t neg_gl64(gl64_t x)
{
    uint64_t v = x.get_val();
    if (v == 0) {
        return x;
    }
    return gl64_t(gl64_t::MOD - v);
}

// a^(p-2) mod p (Fermat); works for cpp_gl64_t and device gl64_t.
static GL64_EXT2_INLINE gl64_t inv_base(gl64_t a)
{
    uint64_t e = gl64_t::MOD - 2;
    gl64_t r = gl64_t::one();
    gl64_t b = a;
    while (e > 0) {
        if (e & 1ULL) {
            r = r * b;
        }
        b = b * b;
        e >>= 1;
    }
    return r;
}

// Goldilocks quadratic extension: F_p[x] / (x^2 - 7), p = 2^64 - 2^32 + 1.
// Element = real + imag * x with x^2 = 7 (Plonky2 `QuadraticExtension<GoldilocksField>`).
struct gl64_ext2_t {
    gl64_t real;
    gl64_t imag;

    // W = 7 (non-residue defining the extension).
    static GL64_EXT2_HD gl64_t w() { return gl64_t(7); }

    // Plonky2 `Extendable<2>::EXT_POWER_OF_TWO_GENERATOR` (see `goldilocks_extensions.rs`).
    static constexpr uint64_t EXT_POWER_OF_TWO_GENERATOR_IMAG = 15659105665374529263ULL;
    // Extension two-adicity = base TWO_ADICITY (32) + 1.
    static constexpr size_t EXT_TWO_ADICITY = 33;

    GL64_EXT2_HD gl64_ext2_t() : real(gl64_t::zero()), imag(gl64_t::zero()) {}

    GL64_EXT2_HD gl64_ext2_t(gl64_t r, gl64_t i) : real(r), imag(i) {}

    GL64_EXT2_HD gl64_ext2_t(gl64_t r) : real(r), imag(gl64_t::zero()) {}

    GL64_EXT2_HD gl64_ext2_t(uint64_t r) : real(gl64_t(r)), imag(gl64_t::zero()) {}

    static GL64_EXT2_HD gl64_ext2_t zero()
    {
        return gl64_ext2_t();
    }

    static GL64_EXT2_HD gl64_ext2_t one()
    {
        return gl64_ext2_t(gl64_t::one(), gl64_t::zero());
    }

    friend GL64_EXT2_INLINE gl64_ext2_t operator+(gl64_ext2_t a, const gl64_ext2_t &b)
    {
        return gl64_ext2_t(a.real + b.real, a.imag + b.imag);
    }

    friend GL64_EXT2_INLINE gl64_ext2_t operator-(gl64_ext2_t a, const gl64_ext2_t &b)
    {
        return gl64_ext2_t(a.real - b.real, a.imag - b.imag);
    }

    friend GL64_EXT2_INLINE gl64_ext2_t operator*(const gl64_ext2_t &a, const gl64_ext2_t &b)
    {
        gl64_t t0 = a.real * b.real;
        gl64_t t1 = a.imag * b.imag;
        gl64_t t2 = (a.real * b.imag) + (a.imag * b.real);
        gl64_t wbd = w() * t1;
        return gl64_ext2_t(t0 + wbd, t2);
    }

    friend GL64_EXT2_INLINE gl64_ext2_t operator*(gl64_t scalar, const gl64_ext2_t &a)
    {
        return gl64_ext2_t(scalar * a.real, scalar * a.imag);
    }

    static GL64_EXT2_HD gl64_ext2_t neg(const gl64_ext2_t &a)
    {
        return gl64_ext2_t(neg_gl64(a.real), neg_gl64(a.imag));
    }

    // Norm N(a) = a0^2 - 7*a1^2  (base field).
    GL64_EXT2_HD gl64_t norm() const
    {
        return (real * real) - (w() * (imag * imag));
    }

    static GL64_EXT2_HD gl64_ext2_t inverse(const gl64_ext2_t &a)
    {
        gl64_t n = a.norm();
        gl64_t n_inv = inv_base(n);
        // (-a1) * n_inv  (use canonical negation; cpp_gl64_t's zero()-x can differ by EPSILON in edge cases)
        gl64_t imag_inv = neg_gl64(a.imag) * n_inv;
        return gl64_ext2_t(a.real * n_inv, imag_inv);
    }

    GL64_EXT2_HD gl64_ext2_t pow(uint64_t exp) const
    {
        gl64_ext2_t res = one();
        gl64_ext2_t base = *this;
        uint64_t ee = exp;
        while (ee > 0) {
            if (ee & 1ULL) {
                res = res * base;
            }
            base = base * base;
            ee >>= 1;
        }
        return res;
    }

    GL64_EXT2_HD gl64_ext2_t exp_power_of_2(size_t power_log) const
    {
        gl64_ext2_t res = *this;
        for (size_t i = 0; i < power_log; ++i) {
            res = res * res;
        }
        return res;
    }

    // `primitive_root_of_unity(n_log)` for the extension field (Plonky2 `Field::primitive_root_of_unity`).
    static GL64_EXT2_HD gl64_ext2_t primitive_root_of_unity(size_t n_log)
    {
        assert(n_log <= EXT_TWO_ADICITY);
        gl64_ext2_t base(gl64_t::zero(), gl64_t(EXT_POWER_OF_TWO_GENERATOR_IMAG));
        return base.exp_power_of_2(EXT_TWO_ADICITY - n_log);
    }

    friend GL64_EXT2_INLINE bool operator==(const gl64_ext2_t &a, const gl64_ext2_t &b)
    {
        return (a.real.get_val() == b.real.get_val()) && (a.imag.get_val() == b.imag.get_val());
    }

    friend GL64_EXT2_INLINE bool operator!=(const gl64_ext2_t &a, const gl64_ext2_t &b)
    {
        return !(a == b);
    }
};

#undef GL64_EXT2_HD
#undef GL64_EXT2_INLINE

#endif // __ZEKNOX_PROVER_GL64_EXT2_CUH__
