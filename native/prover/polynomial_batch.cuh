// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_POLYNOMIAL_BATCH_CUH__
#define __ZEKNOX_PROVER_POLYNOMIAL_BATCH_CUH__

#include <cstddef>
#include <cstdint>
#include <utils/rusterror.h>
#include <merkle/merkle.h>

#ifdef __cplusplus

#include <ff/goldilocks.hpp>
#include <utils/gpu_t.cuh>
#include <ntt/ntt.h>

static constexpr size_t SALT_SIZE = 4;
static constexpr size_t NUM_HASH_OUT_ELTS = 4;

class PolynomialBatchGPU {
public:
    fr_t *lde_gpu;
    size_t num_leaves;
    size_t leaf_size;

    fr_t *digests_gpu;
    fr_t *cap_gpu;
    size_t num_digests;
    size_t cap_len;

    fr_t *coeffs_gpu;
    size_t num_polynomials;

    size_t degree_log;
    size_t rate_bits;
    bool blinding;
    size_t cap_height;
    size_t gpu_id;

    bool owns_coeffs;

    PolynomialBatchGPU();
    ~PolynomialBatchGPU();

    PolynomialBatchGPU(const PolynomialBatchGPU &) = delete;
    PolynomialBatchGPU &operator=(const PolynomialBatchGPU &) = delete;

    PolynomialBatchGPU(PolynomialBatchGPU &&other) noexcept;
    PolynomialBatchGPU &operator=(PolynomialBatchGPU &&other) noexcept;

    static PolynomialBatchGPU from_values(
        fr_t *values_gpu,
        size_t num_polys,
        size_t degree_log,
        size_t rate_bits,
        bool blinding,
        size_t cap_height,
        size_t gpu_id
    );

    static PolynomialBatchGPU from_coeffs(
        fr_t *coeffs_gpu,
        size_t num_polys,
        size_t degree_log,
        size_t rate_bits,
        bool blinding,
        size_t cap_height,
        size_t gpu_id
    );

    size_t lde_values_offset(size_t index, size_t step) const;

    void copy_cap_to_host(fr_t *host_cap, size_t max_elems) const;

    void copy_digests_to_host(fr_t *host_digests, size_t max_elems) const;

private:
    void build_lde_and_merkle(size_t num_polys, size_t degree_log,
                              size_t rate_bits, bool blinding,
                              size_t cap_height, size_t gpu_id);
};

#endif // __cplusplus

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C RustError polynomial_batch_from_values(
    size_t device_id,
    const void *values_gpu,
    uint32_t num_polys,
    uint32_t degree_log,
    uint32_t rate_bits,
    int blinding,
    uint32_t cap_height,
    void **out_lde_gpu,
    void **out_digests_gpu,
    void **out_cap_gpu,
    void **out_coeffs_gpu,
    uint64_t *out_num_leaves,
    uint64_t *out_leaf_size,
    uint64_t *out_num_digests,
    uint64_t *out_cap_len
);

EXTERN_C RustError polynomial_batch_from_coeffs(
    size_t device_id,
    const void *coeffs_gpu,
    uint32_t num_polys,
    uint32_t degree_log,
    uint32_t rate_bits,
    int blinding,
    uint32_t cap_height,
    void **out_lde_gpu,
    void **out_digests_gpu,
    void **out_cap_gpu,
    uint64_t *out_num_leaves,
    uint64_t *out_leaf_size,
    uint64_t *out_num_digests,
    uint64_t *out_cap_len
);

#undef EXTERN_C

#endif // __ZEKNOX_PROVER_POLYNOMIAL_BATCH_CUH__
