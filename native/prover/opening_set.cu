// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#include <utils/all_gpus.hpp>
#include <utils/gpu_t.cuh>
#include <utils/exception.cuh>
#include <ff/goldilocks.hpp>

#include "prover/opening_set.cuh"

namespace {

constexpr unsigned kEvalBlock = 256;

/// One thread per polynomial: Horner in `F_p[x]/(x^2-7)` with base-field coefficients.
__global__ void eval_polynomials_at_point_kernel(
    const fr_t *coeffs_gpu,
    size_t poly_offset,
    size_t num_polys,
    size_t degree,
    fr_t zeta_real,
    fr_t zeta_imag,
    fr_t *results)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_polys) {
        return;
    }
    gl64_ext2_t z(zeta_real, zeta_imag);
    const fr_t *coeffs = coeffs_gpu + (poly_offset + tid) * degree;

    gl64_ext2_t acc = gl64_ext2_t(coeffs[degree - 1], fr_t::zero());
    for (size_t idx = degree - 1; idx > 0; idx--) {
        acc = acc * z + gl64_ext2_t(coeffs[idx - 1], fr_t::zero());
    }

    size_t o = tid * 2;
    results[o] = acc.real;
    results[o + 1] = acc.imag;
}

static void launch_eval_polynomials_at_point(
    const fr_t *coeffs_gpu,
    size_t poly_offset,
    size_t num_polys,
    size_t degree,
    gl64_ext2_t zeta,
    fr_t *results_gpu,
    cudaStream_t stream)
{
    if (num_polys == 0) {
        return;
    }
    dim3 block(kEvalBlock);
    dim3 grid((num_polys + kEvalBlock - 1) / kEvalBlock);
    eval_polynomials_at_point_kernel<<<grid, block, 0, stream>>>(
        coeffs_gpu, poly_offset, num_polys, degree, zeta.real, zeta.imag, results_gpu);
    CUDA_OK(cudaGetLastError());
}

static void range_check(const char *name, size_t start, size_t end, size_t batch_n)
{
    if (start > end || end > batch_n) {
        throw cuda_error{-cudaErrorInvalidValue,
                         std::string("opening_set: invalid range for ") + name};
    }
}

static void assert_same_degree_log(
    const PolynomialBatchGPU &a,
    const PolynomialBatchGPU &b,
    const char *a_name,
    const char *b_name)
{
    if (a.degree_log != b.degree_log) {
        throw cuda_error{-cudaErrorInvalidValue,
                         std::string("opening_set: degree_log mismatch between ") + a_name +
                             " and " + b_name};
    }
}

static std::vector<gl64_ext2_t> eval_contiguous(
    const PolynomialBatchGPU &batch,
    size_t poly_start,
    size_t poly_count,
    gl64_ext2_t point,
    fr_t *d_scratch,
    const gpu_t &gpu)
{
    if (poly_count == 0) {
        return {};
    }
    if (batch.coeffs_gpu == nullptr) {
        throw cuda_error{-cudaErrorInvalidValue, "opening_set: coeffs_gpu is null"};
    }
    size_t degree = (size_t)1 << batch.degree_log;

    cudaStream_t stream = static_cast<cudaStream_t>(gpu);
    launch_eval_polynomials_at_point(
        batch.coeffs_gpu, poly_start, poly_count, degree, point, d_scratch, stream);

    std::vector<fr_t> host_flat(poly_count * 2);
    CUDA_OK(cudaMemcpy(host_flat.data(), d_scratch, poly_count * 2 * sizeof(fr_t), cudaMemcpyDeviceToHost));

    std::vector<gl64_ext2_t> out;
    out.reserve(poly_count);
    for (size_t i = 0; i < poly_count; i++) {
        out.emplace_back(host_flat[i * 2], host_flat[i * 2 + 1]);
    }
    return out;
}

} // namespace

OpeningSet construct_opening_set(
    gl64_ext2_t zeta,
    gl64_ext2_t g,
    const PolynomialBatchGPU &constants_sigmas,
    const PolynomialBatchGPU &wires,
    const PolynomialBatchGPU &zs_partial_products,
    const PolynomialBatchGPU &quotient_polys,
    size_t constants_start,
    size_t constants_end,
    size_t sigmas_start,
    size_t sigmas_end,
    size_t zs_start,
    size_t zs_end,
    size_t partial_products_start,
    size_t partial_products_end,
    size_t gpu_id)
{
    assert_same_degree_log(constants_sigmas, wires, "constants_sigmas", "wires");
    assert_same_degree_log(constants_sigmas, zs_partial_products, "constants_sigmas", "zs_partial_products");
    assert_same_degree_log(constants_sigmas, quotient_polys, "constants_sigmas", "quotient_polys");

    range_check("constants", constants_start, constants_end, constants_sigmas.num_polynomials);
    range_check("sigmas", sigmas_start, sigmas_end, constants_sigmas.num_polynomials);
    range_check("zs", zs_start, zs_end, zs_partial_products.num_polynomials);
    range_check("partial_products", partial_products_start, partial_products_end,
                zs_partial_products.num_polynomials);

    // Find the largest poly count across all eval calls to size a single scratch buffer.
    size_t max_polys = std::max({
        constants_end - constants_start,
        sigmas_end - sigmas_start,
        wires.num_polynomials,
        zs_end - zs_start,
        partial_products_end - partial_products_start,
        quotient_polys.num_polynomials
    });

    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();

    auto cuda_deleter = [](fr_t* ptr) { if (ptr) cudaFree(ptr); };
    std::unique_ptr<fr_t, decltype(cuda_deleter)> d_scratch_ptr(nullptr, cuda_deleter);
    if (max_polys > 0) {
        fr_t* raw_ptr;
        CUDA_OK(cudaMalloc(&raw_ptr, max_polys * 2 * sizeof(fr_t)));
        d_scratch_ptr.reset(raw_ptr);
    }
    fr_t *d_scratch = d_scratch_ptr.get();

    gl64_ext2_t gz = g * zeta;

    OpeningSet out;
    out.constants = eval_contiguous(constants_sigmas, constants_start,
                                    constants_end - constants_start, zeta, d_scratch, gpu);
    out.plonk_sigmas = eval_contiguous(constants_sigmas, sigmas_start,
                                       sigmas_end - sigmas_start, zeta, d_scratch, gpu);
    out.wires = eval_contiguous(wires, 0, wires.num_polynomials, zeta, d_scratch, gpu);
    out.plonk_zs = eval_contiguous(zs_partial_products, zs_start,
                                    zs_end - zs_start, zeta, d_scratch, gpu);
    out.partial_products =
        eval_contiguous(zs_partial_products, partial_products_start,
                        partial_products_end - partial_products_start, zeta, d_scratch, gpu);
    out.plonk_zs_next = eval_contiguous(zs_partial_products, zs_start,
                                         zs_end - zs_start, gz, d_scratch, gpu);
    out.quotient_polys = eval_contiguous(quotient_polys, 0,
                                          quotient_polys.num_polynomials, zeta, d_scratch, gpu);
    return out;
}
