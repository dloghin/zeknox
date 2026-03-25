// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <ff/gl64_params.hpp>
#include <ff/goldilocks.hpp>
#include <merkle/hasher.hpp>
#include <ntt/ntt.cuh>
#include <ntt/ntt.h>
#include <prover/challenger.hpp>
#include <prover/gl64_ext2.cuh>
#include <prover/opening_set.cuh>
#include <prover/partial_products.cuh>
#include <prover/polynomial_batch.cuh>
#include <prover/prover.h>
#include <prover/gate_constraints.cuh>
#include <prover/quotient_compute_polys.cuh>
#include <prover/quotient_poly.cuh>
#include <utils/all_gpus.hpp>
#include <utils/exception.cuh>
#include <utils/gpu_t.cuh>

namespace {

constexpr uint32_t kProofMagic = 0x584e4b5a;
constexpr uint32_t kProofVersion = 1u;

struct cuda_fr_deleter {
    cudaStream_t stream{};
    void operator()(fr_t *p) const noexcept
    {
        if (p) {
            (void)cudaFreeAsync((void *)p, stream);
        }
    }
};

struct cuda_u64_deleter {
    cudaStream_t stream{};
    void operator()(uint64_t *p) const noexcept
    {
        if (p) {
            (void)cudaFreeAsync((void *)p, stream);
        }
    }
};

using unique_fr = std::unique_ptr<fr_t, cuda_fr_deleter>;
using unique_u64 = std::unique_ptr<uint64_t, cuda_u64_deleter>;

static RustError rust_ok()
{
    return RustError{0};
}

static RustError rust_err(int code, const std::string &msg)
{
    return RustError{code, msg};
}

static RustError rust_err_cuda(const zeknox_error &e)
{
    return RustError{e.code(), e.what()};
}

static RustError rust_err_std(const std::exception &e)
{
    return RustError{-1, e.what()};
}

__global__ void transpose_trace_poly_to_row_kernel(
    const fr_t *poly_major,
    uint64_t *row_major,
    size_t degree,
    size_t num_wires)
{
    size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t total = degree * num_wires;
    if (tid >= total) {
        return;
    }
    size_t row = tid / num_wires;
    size_t w = tid % num_wires;
    row_major[tid] = poly_major[w * degree + row].get_val();
}

static void launch_transpose_trace(
    const fr_t *poly_major,
    uint64_t *row_major,
    size_t degree,
    size_t num_wires,
    cudaStream_t stream)
{
    if (degree == 0 || num_wires == 0) {
        return;
    }
    size_t total = degree * num_wires;
    int threads = 256;
    int blocks = (int)((total + (size_t)threads - 1) / (size_t)threads);
    transpose_trace_poly_to_row_kernel<<<blocks, threads, 0, stream>>>(
        poly_major, row_major, degree, num_wires);
    CUDA_OK(cudaGetLastError());
}

static void flatten_opening_set_for_transcript(const OpeningSet &os, std::vector<gl64_t> &flat)
{
    flat.clear();
    auto push_ext = [&](const gl64_ext2_t &x) {
        flat.push_back(x.real);
        flat.push_back(x.imag);
    };
    for (const auto &x : os.constants) {
        push_ext(x);
    }
    for (const auto &x : os.plonk_sigmas) {
        push_ext(x);
    }
    for (const auto &x : os.wires) {
        push_ext(x);
    }
    for (const auto &x : os.plonk_zs) {
        push_ext(x);
    }
    for (const auto &x : os.plonk_zs_next) {
        push_ext(x);
    }
    for (const auto &x : os.partial_products) {
        push_ext(x);
    }
    for (const auto &x : os.quotient_polys) {
        push_ext(x);
    }
}

static size_t proof_bytes_needed(
    size_t cap_elems_wire,
    size_t cap_elems_zp,
    size_t cap_elems_q,
    size_t opening_gl64_count)
{
    return sizeof(uint32_t) * 3 + sizeof(uint64_t) * 3 +
           cap_elems_wire * sizeof(uint64_t) + cap_elems_zp * sizeof(uint64_t) +
           cap_elems_q * sizeof(uint64_t) + sizeof(uint64_t) +
           opening_gl64_count * sizeof(uint64_t);
}

static void write_u32(uint8_t *&p, uint32_t v)
{
    std::memcpy(p, &v, sizeof(v));
    p += sizeof(v);
}

static void write_u64(uint8_t *&p, uint64_t v)
{
    std::memcpy(p, &v, sizeof(v));
    p += sizeof(v);
}

/** Twiddle + coset tables for NTT/LDE (matches tests `init_gpu_for_poly_batch`). */
static void init_ntt_for_prove(uint32_t max_lg_domain, uint64_t gpu_id)
{
    auto &gpu = select_gpu((int)gpu_id);
    gpu.select();
    RustError ce = ntt::init_coset(gpu, max_lg_domain, fr_t(GROUP_GENERATOR));
    if (ce.code != 0) {
        throw cuda_error{ce.code, ce.message ? ce.message : "init_coset failed"};
    }
    for (uint32_t k = 2; k <= max_lg_domain; k++) {
        RustError te = ntt::init_twiddle_factors(gpu, k);
        if (te.code != 0) {
            throw cuda_error{te.code, te.message ? te.message : "init_twiddle_factors failed"};
        }
    }
}

} // namespace

RustError gpu_prove(
    const void *constants_sigmas_coeffs_gpu,
    const void *constants_sigmas_lde_gpu,
    const void *constants_sigmas_digests_gpu,
    const void *constants_sigmas_cap_gpu,
    const void *circuit_digest,
    const void *public_inputs_hash,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const void *k_is,
    const void *subgroup,
    const void *wire_values,
    const uint32_t *reduction_arity_bits,
    uint32_t num_fri_rounds,
    const void *fft_root_table,
    void *proof_output,
    size_t *proof_size,
    uint64_t gpu_id)
{
    (void)constants_sigmas_lde_gpu;
    (void)constants_sigmas_digests_gpu;
    (void)constants_sigmas_cap_gpu;
    (void)gates;
    (void)num_gates;
    (void)reduction_arity_bits;
    (void)fft_root_table;

    if (!proof_size) {
        return rust_err(EINVAL, "gpu_prove: proof_size is null");
    }
    if (!proof_output) {
        return rust_err(EINVAL, "gpu_prove: proof_output is null");
    }
    if (!config) {
        return rust_err(EINVAL, "gpu_prove: config is null");
    }
    if (!constants_sigmas_coeffs_gpu || !circuit_digest || !public_inputs_hash || !k_is || !subgroup ||
        !wire_values) {
        return rust_err(EINVAL, "gpu_prove: required pointer argument is null");
    }
    if (config->degree_bits == 0 || config->degree_bits > 31) {
        return rust_err(EINVAL, "gpu_prove: invalid degree_bits");
    }
    if (config->num_wires == 0) {
        return rust_err(EINVAL, "gpu_prove: num_wires is zero");
    }
    if (config->num_routed_wires == 0) {
        return rust_err(EINVAL, "gpu_prove: num_routed_wires is zero (permutation argument required)");
    }
    if (config->quotient_degree_factor == 0) {
        return rust_err(EINVAL, "gpu_prove: quotient_degree_factor is zero");
    }
    if (num_fri_rounds != 0) {
        return rust_err(EINVAL,
                        "gpu_prove: num_fri_rounds > 0 (FRI composition wiring not implemented in this MVP)");
    }

    const size_t degree = (size_t)1u << config->degree_bits;
    const size_t num_const_sigma = (size_t)config->num_constants + (size_t)config->num_routed_wires;
    const size_t num_chunks = partial_products_num_chunks(config->num_routed_wires, config->quotient_degree_factor);
    const size_t num_zp_polys = 1u + num_chunks;
    const uint32_t qfac = std::max(1u, config->quotient_degree_factor);
    // Plonky2 expects `num_challenges * quotient_degree_factor` quotient chunks.
    const size_t num_q_polys = (size_t)config->num_challenges * (size_t)qfac;

    try {
        init_ntt_for_prove(config->degree_bits + config->rate_bits + 4u, gpu_id);

        auto &gpu = select_gpu((int)gpu_id);
        gpu.select();
        cudaStream_t stream = static_cast<cudaStream_t>(gpu);

        const gl64_t *digest_in = static_cast<const gl64_t *>(circuit_digest);
        const gl64_t *pub_in = static_cast<const gl64_t *>(public_inputs_hash);

        fr_t *d_wire_raw = nullptr;
        CUDA_OK(cudaMalloc(&d_wire_raw, degree * config->num_wires * sizeof(fr_t)));
        unique_fr d_wire_poly(d_wire_raw, cuda_fr_deleter{stream});
        CUDA_OK(cudaMemcpy(
            d_wire_poly.get(),
            wire_values,
            degree * config->num_wires * sizeof(fr_t),
            cudaMemcpyHostToDevice));

        uint64_t *d_wr = nullptr;
        CUDA_OK(cudaMalloc(&d_wr, degree * config->num_wires * sizeof(uint64_t)));
        unique_u64 d_wire_row(d_wr, cuda_u64_deleter{stream});
        launch_transpose_trace(d_wire_poly.get(), d_wire_row.get(), degree, config->num_wires, stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        PolynomialBatchGPU wires_batch = PolynomialBatchGPU::from_values(
            d_wire_poly.get(),
            config->num_wires,
            config->degree_bits,
            config->rate_bits,
            false,
            config->cap_height,
            (size_t)gpu_id);

        Challenger challenger;
        challenger.observe_hash(digest_in);
        challenger.observe_hash(pub_in);
        {
            std::vector<gl64_t> cap_host(wires_batch.cap_len * NUM_HASH_OUT_ELTS);
            wires_batch.copy_cap_to_host(cap_host.data(), cap_host.size());
            challenger.observe_cap(cap_host.data(), cap_host.size());
        }

        gl64_t beta = challenger.get_challenge();
        gl64_t gamma = challenger.get_challenge();

        fr_t *d_sig = nullptr;
        CUDA_OK(cudaMalloc(&d_sig, degree * config->num_routed_wires * sizeof(fr_t)));
        unique_fr d_sigma_trace(d_sig, cuda_fr_deleter{stream});
        CUDA_OK(cudaMemcpy(
            d_sigma_trace.get(),
            static_cast<const fr_t *>(constants_sigmas_coeffs_gpu) +
                (size_t)config->num_constants * degree,
            degree * config->num_routed_wires * sizeof(fr_t),
            cudaMemcpyDeviceToDevice));

        {
            NTT_Config ntt_cfg = {};
            ntt_cfg.batches = config->num_routed_wires;
            ntt_cfg.order = NN;
            ntt_cfg.ntt_type = standard;
            ntt_cfg.extension_rate_bits = 0;
            ntt_cfg.are_inputs_on_device = true;
            ntt_cfg.are_outputs_on_device = true;
            ntt_cfg.with_coset = false;
            ntt_cfg.is_multi_gpu = false;
            ntt_cfg.salt_size = 0;
            RustError nerr =
                ntt::batch_ntt(gpu, d_sigma_trace.get(), (uint32_t)config->degree_bits, forward, ntt_cfg);
            if (nerr.code != 0) {
                return RustError{nerr.code, nerr.message ? nerr.message : "sigma forward NTT failed"};
            }
        }

        // Transpose sigma from polynomial-major (after NTT) to row-major for partial products kernel
        uint64_t *d_sr = nullptr;
        CUDA_OK(cudaMalloc(&d_sr, degree * config->num_routed_wires * sizeof(uint64_t)));
        unique_u64 d_sigma_row(d_sr, cuda_u64_deleter{stream});
        launch_transpose_trace(
            d_sigma_trace.get(), d_sigma_row.get(), degree, config->num_routed_wires, stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        d_sigma_trace.reset();

        uint64_t *d_k_raw = nullptr;
        uint64_t *d_su_raw = nullptr;
        CUDA_OK(cudaMalloc(&d_k_raw, config->num_routed_wires * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc(&d_su_raw, degree * sizeof(uint64_t)));
        unique_u64 d_k(d_k_raw, cuda_u64_deleter{stream});
        unique_u64 d_sub(d_su_raw, cuda_u64_deleter{stream});
        CUDA_OK(cudaMemcpy(
            d_k.get(), k_is, config->num_routed_wires * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_sub.get(), subgroup, degree * sizeof(uint64_t), cudaMemcpyHostToDevice));

        uint64_t *d_ch = nullptr;
        CUDA_OK(cudaMalloc(&d_ch, degree * num_chunks * sizeof(uint64_t)));
        unique_u64 d_chunk(d_ch, cuda_u64_deleter{stream});
        launch_compute_quotient_chunk_products(
            d_wire_row.get(),
            d_sigma_row.get(),
            d_sub.get(),
            d_k.get(),
            beta.get_val(),
            gamma.get_val(),
            degree,
            config->num_wires,
            config->num_routed_wires,
            config->quotient_degree_factor,
            d_chunk.get(),
            stream);
        CUDA_OK(cudaStreamSynchronize(stream));

        d_sigma_row.reset();
        d_k.reset();
        d_sub.reset();
        d_wire_row.reset();

        uint64_t *d_pa = nullptr;
        uint64_t *d_zz = nullptr;
        CUDA_OK(cudaMalloc(&d_pa, degree * num_chunks * sizeof(uint64_t)));
        CUDA_OK(cudaMalloc(&d_zz, degree * sizeof(uint64_t)));
        unique_u64 d_partial(d_pa, cuda_u64_deleter{stream});
        unique_u64 d_z(d_zz, cuda_u64_deleter{stream});
        launch_compute_z_prefix_product_gpu(
            d_chunk.get(), degree, num_chunks, d_partial.get(), d_z.get(), stream);
        d_chunk.reset();

        fr_t *d_zpr = nullptr;
        CUDA_OK(cudaMalloc(&d_zpr, num_zp_polys * degree * sizeof(fr_t)));
        unique_fr d_zp_vals(d_zpr, cuda_fr_deleter{stream});
        launch_pack_zs_pp_polynomials(
            d_z.get(),
            d_partial.get(),
            reinterpret_cast<uint64_t *>(d_zp_vals.get()),
            degree,
            num_chunks,
            stream);
        CUDA_OK(cudaStreamSynchronize(stream));
        d_z.reset();
        d_partial.reset();

        PolynomialBatchGPU zp_batch = PolynomialBatchGPU::from_values(
            d_zp_vals.get(),
            num_zp_polys,
            config->degree_bits,
            config->rate_bits,
            false,
            config->cap_height,
            (size_t)gpu_id);

        {
            std::vector<gl64_t> cap_host(zp_batch.cap_len * NUM_HASH_OUT_ELTS);
            zp_batch.copy_cap_to_host(cap_host.data(), cap_host.size());
            challenger.observe_cap(cap_host.data(), cap_host.size());
        }

        PolynomialBatchGPU cs_batch = PolynomialBatchGPU::from_coeffs(
            static_cast<fr_t *>(const_cast<void *>(constants_sigmas_coeffs_gpu)),
            num_const_sigma,
            config->degree_bits,
            config->rate_bits,
            false,
            config->cap_height,
            (size_t)gpu_id);

        /* Alpha challenges for quotient mixing (Plonky2 transcript). */
        std::vector<gl64_t> alphas = challenger.get_n_challenges(config->num_challenges);

        unique_u64 d_qvals(nullptr, cuda_u64_deleter{stream});
        bool have_qvals = false;
        bool quotient_ifft_done = false;
        size_t lde_q_stride = 0;

        const bool use_plonky2_quotient =
            config->num_challenges == 1u && config->quotient_degree_factor >= 2u &&
            num_zp_polys == 1u + num_chunks &&
            (config->num_gate_constraints == 0u || num_gates > 0u);

        if (use_plonky2_quotient) {
            std::vector<uint64_t> betas_u64(1, beta.get_val());
            std::vector<uint64_t> gammas_u64(1, gamma.get_val());
            std::vector<uint64_t> alphas_u64(1, alphas[0].get_val());
            std::vector<uint64_t> k_is_host((size_t)config->num_routed_wires);
            memcpy(k_is_host.data(), k_is, k_is_host.size() * sizeof(uint64_t));

            uint32_t q_bits = config->quotient_degree_factor <= 1u
                                  ? 0u
                                  : (32u - (uint32_t)__builtin_clz(config->quotient_degree_factor - 1u));
            size_t lde_q_est = (size_t)1u << ((size_t)config->degree_bits + (size_t)q_bits);
            uint64_t *d_qfull = nullptr;
            CUDA_OK(cudaMalloc(
                &d_qfull, (size_t)config->num_challenges * lde_q_est * sizeof(uint64_t)));

            RustError qerr = compute_quotient_polys_from_batches_gl64(
                (size_t)gpu_id,
                stream,
                cs_batch,
                wires_batch,
                zp_batch,
                config,
                gates,
                num_gates,
                k_is_host.data(),
                reinterpret_cast<const uint64_t *>(pub_in),
                betas_u64.data(),
                gammas_u64.data(),
                alphas_u64.data(),
                reinterpret_cast<fr_t *>(d_qfull),
                &lde_q_stride);
            if (qerr.code != 0) {
                cudaFree(d_qfull);
                return qerr;
            }
            d_qvals = unique_u64(d_qfull, cuda_u64_deleter{stream});
            have_qvals = true;
            quotient_ifft_done = true;
        } else if (config->num_gate_constraints > 0 && num_gates > 0) {
            const size_t lde_log = (size_t)config->degree_bits + (size_t)config->rate_bits;
            const size_t lde_size = (size_t)1u << lde_log;
            const size_t n_gc = (size_t)config->num_gate_constraints;

            uint64_t *d_gc_raw = nullptr;
            uint64_t *d_alpha_raw = nullptr;
            uint64_t *d_zh_inv_raw = nullptr;
            uint64_t *d_qvals_raw = nullptr;
            CUDA_OK(cudaMalloc(&d_gc_raw, n_gc * lde_size * sizeof(uint64_t)));
            CUDA_OK(cudaMalloc(&d_alpha_raw, std::max((size_t)1, n_gc) * sizeof(uint64_t)));
            CUDA_OK(cudaMalloc(&d_zh_inv_raw, lde_size * sizeof(uint64_t)));
            CUDA_OK(cudaMalloc(&d_qvals_raw, (size_t)config->num_challenges * lde_size * sizeof(uint64_t)));
            unique_u64 d_gc(d_gc_raw, cuda_u64_deleter{stream});
            unique_u64 d_alpha(d_alpha_raw, cuda_u64_deleter{stream});
            unique_u64 d_zh_inv(d_zh_inv_raw, cuda_u64_deleter{stream});
            d_qvals = unique_u64(d_qvals_raw, cuda_u64_deleter{stream});
            CUDA_OK(cudaMemset(d_gc.get(), 0, n_gc * lde_size * sizeof(uint64_t)));

            std::vector<uint64_t> alpha_host(std::max((size_t)1, n_gc), gl64_t::one().get_val());
            for (size_t i = 0; i < n_gc; i++) {
                alpha_host[i] = alphas[i % std::max((size_t)1, alphas.size())].get_val();
            }
            CUDA_OK(cudaMemcpy(
                d_alpha.get(),
                alpha_host.data(),
                alpha_host.size() * sizeof(uint64_t),
                cudaMemcpyHostToDevice));

            size_t row = 0;
            for (size_t gi = 0; gi < (size_t)num_gates && row < n_gc; ++gi) {
                const GateInfo &g = gates[gi];
                if (g.gate_type == (uint32_t)GateType::ArithmeticGate) {
                    const uint32_t nops = std::max(1u, g.aux_0);
                    for (uint32_t op = 0; op < nops && row < n_gc; ++op, ++row) {
                        launch_eval_arithmetic_gate_constraints(
                            reinterpret_cast<const uint64_t *>(cs_batch.lde_gpu),
                            reinterpret_cast<const uint64_t *>(wires_batch.lde_gpu),
                            lde_size,
                            (size_t)config->num_constants,
                            (size_t)config->num_wires,
                            g.wire_0 + 4u * op,
                            g.wire_1 + 4u * op,
                            g.wire_2 + 4u * op,
                            g.wire_3 + 4u * op,
                            g.const_0,
                            g.const_1,
                            row,
                            d_gc.get(),
                            n_gc,
                            stream);
                    }
                } else if (g.gate_type == (uint32_t)GateType::ConstantGate) {
                    const uint32_t nconst = std::max(1u, g.aux_0);
                    for (uint32_t j = 0; j < nconst && row < n_gc; ++j, ++row) {
                        launch_eval_constant_gate_constraints(
                            reinterpret_cast<const uint64_t *>(cs_batch.lde_gpu),
                            reinterpret_cast<const uint64_t *>(wires_batch.lde_gpu),
                            lde_size,
                            (size_t)config->num_constants,
                            (size_t)config->num_wires,
                            g.wire_0 + j,
                            g.const_0 + j,
                            row,
                            d_gc.get(),
                            n_gc,
                            stream);
                    }
                }
            }

            launch_precompute_z_h_inverse(
                (uint64_t)GROUP_GENERATOR,
                (uint64_t)OMEGA[lde_log],
                config->degree_bits,
                d_zh_inv.get(),
                lde_size,
                stream);

            launch_eval_vanishing_poly(
                d_gc.get(),
                n_gc,
                lde_size,
                d_alpha.get(),
                n_gc,
                nullptr,
                0,
                d_zh_inv.get(),
                d_qvals.get(),
                config->num_challenges,
                stream);
            CUDA_OK(cudaStreamSynchronize(stream));
            have_qvals = true;
            lde_q_stride = lde_size;
        }

        fr_t *d_qc = nullptr;
        CUDA_OK(cudaMalloc(&d_qc, num_q_polys * degree * sizeof(fr_t)));
        unique_fr d_q_coeffs(d_qc, cuda_fr_deleter{stream});
        CUDA_OK(cudaMemset(d_q_coeffs.get(), 0, num_q_polys * degree * sizeof(fr_t)));
        if (have_qvals) {
            if (!quotient_ifft_done) {
                const size_t lde_log = (size_t)config->degree_bits + (size_t)config->rate_bits;
                NTT_Config inv_cfg = {};
                inv_cfg.batches = config->num_challenges;
                inv_cfg.order = NN;
                inv_cfg.ntt_type = standard;
                inv_cfg.extension_rate_bits = 0;
                inv_cfg.are_inputs_on_device = true;
                inv_cfg.are_outputs_on_device = true;
                inv_cfg.with_coset = true;
                inv_cfg.is_multi_gpu = false;
                inv_cfg.salt_size = 0;
                RustError ierr = ntt::batch_ntt(
                    gpu, reinterpret_cast<fr_t *>(d_qvals.get()), (uint32_t)lde_log, inverse, inv_cfg);
                if (ierr.code != 0) {
                    return RustError{ierr.code, ierr.message ? ierr.message : "quotient inverse NTT failed"};
                }
            }

            const size_t per_ch = (size_t)qfac * degree;
            for (size_t ch = 0; ch < (size_t)config->num_challenges; ch++) {
                const fr_t *src = reinterpret_cast<fr_t *>(d_qvals.get()) + ch * lde_q_stride;
                fr_t *dst = d_q_coeffs.get() + ch * per_ch;
                CUDA_OK(cudaMemcpyAsync(
                    dst,
                    src,
                    per_ch * sizeof(fr_t),
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
            CUDA_OK(cudaStreamSynchronize(stream));
        }

        PolynomialBatchGPU q_batch = PolynomialBatchGPU::from_coeffs(
            d_q_coeffs.get(),
            num_q_polys,
            config->degree_bits,
            config->rate_bits,
            false,
            config->cap_height,
            (size_t)gpu_id);

        {
            std::vector<gl64_t> cap_host(q_batch.cap_len * NUM_HASH_OUT_ELTS);
            q_batch.copy_cap_to_host(cap_host.data(), cap_host.size());
            challenger.observe_cap(cap_host.data(), cap_host.size());
        }

        gl64_t zeta_ext[2];
        challenger.get_extension_challenge(zeta_ext);
        gl64_ext2_t zeta(zeta_ext[0], zeta_ext[1]);
        gl64_ext2_t zeta_pow_n = zeta.exp_power_of_2(config->degree_bits);
        if (zeta_pow_n == gl64_ext2_t::one()) {
            return rust_err(EINVAL, "gpu_prove: zeta^n == 1 (invalid opening point)");
        }

        gl64_ext2_t g(gl64_t(OMEGA[config->degree_bits]), gl64_t::zero());

        const size_t c0 = 0;
        const size_t c1 = (size_t)config->num_constants;
        const size_t s0 = (size_t)config->num_constants;
        const size_t s1 = (size_t)config->num_constants + (size_t)config->num_routed_wires;
        const size_t zs0 = 0;
        const size_t zs1 = 1;
        const size_t pp0 = 1;
        const size_t pp1 = 1 + num_chunks;

        OpeningSet openings = construct_opening_set(
            zeta,
            g,
            cs_batch,
            wires_batch,
            zp_batch,
            q_batch,
            c0,
            c1,
            s0,
            s1,
            zs0,
            zs1,
            pp0,
            pp1,
            (size_t)gpu_id);

        std::vector<gl64_t> flat_open;
        flatten_opening_set_for_transcript(openings, flat_open);
        challenger.observe_openings(flat_open.data(), flat_open.size());

        const size_t cap_wire_el = wires_batch.cap_len * NUM_HASH_OUT_ELTS;
        const size_t cap_zp_el = zp_batch.cap_len * NUM_HASH_OUT_ELTS;
        const size_t cap_q_el = q_batch.cap_len * NUM_HASH_OUT_ELTS;
        const size_t need = proof_bytes_needed(cap_wire_el, cap_zp_el, cap_q_el, flat_open.size());

        if (*proof_size < need) {
            *proof_size = need;
            /* E2BIG: `proof_output` too small; `*proof_size` holds required byte count (Rust checks libc::E2BIG). */
            return rust_err(E2BIG, "gpu_prove: proof_output buffer too small");
        }

        std::vector<gl64_t> wcap(cap_wire_el);
        std::vector<gl64_t> zcap(cap_zp_el);
        std::vector<gl64_t> qcap(cap_q_el);
        wires_batch.copy_cap_to_host(wcap.data(), wcap.size());
        zp_batch.copy_cap_to_host(zcap.data(), zcap.size());
        q_batch.copy_cap_to_host(qcap.data(), qcap.size());

        uint8_t *out = static_cast<uint8_t *>(proof_output);
        uint8_t *cur = out;
        write_u32(cur, kProofMagic);
        write_u32(cur, kProofVersion);
        write_u32(cur, 1u);
        write_u64(cur, (uint64_t)cap_wire_el);
        for (size_t i = 0; i < cap_wire_el; i++) {
            write_u64(cur, wcap[i].get_val());
        }
        write_u64(cur, (uint64_t)cap_zp_el);
        for (size_t i = 0; i < cap_zp_el; i++) {
            write_u64(cur, zcap[i].get_val());
        }
        write_u64(cur, (uint64_t)cap_q_el);
        for (size_t i = 0; i < cap_q_el; i++) {
            write_u64(cur, qcap[i].get_val());
        }
        write_u64(cur, (uint64_t)flat_open.size());
        for (size_t i = 0; i < flat_open.size(); i++) {
            write_u64(cur, flat_open[i].get_val());
        }

        *proof_size = (size_t)(cur - out);

        return rust_ok();
    } catch (const zeknox_error &e) {
        return rust_err_cuda(e);
    } catch (const std::exception &e) {
        return rust_err_std(e);
    }
}
