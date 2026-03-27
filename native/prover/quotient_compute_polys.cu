// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <ff/gl64_params.hpp>
#include <ff/goldilocks.hpp>
#include <ntt/ntt.cuh>
#include <ntt/ntt.h>
#include <prover/gate_constraints.cuh>
#include <prover/partial_products.cuh>
#include <prover/polynomial_batch.cuh>
#include <prover/gl64_ext2.cuh>
#include <prover/quotient_compute_polys.cuh>
#include <prover/quotient_poly.cuh>
#include <utils/all_gpus.hpp>
#include <utils/exception.cuh>
#include <utils/gpu_t.cuh>

using fr_t = gl64_t;

namespace {

__device__ inline size_t d_rev_bits(size_t val, size_t bit_count)
{
    size_t result = 0;
    for (size_t i = 0; i < bit_count; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

__device__ gl64_t d_gl64_pow_u64(gl64_t base, uint64_t exp)
{
    gl64_t r = gl64_t::one();
    gl64_t b = base;
    while (exp > 0) {
        if (exp & 1ULL) {
            r = r * b;
        }
        b = b * b;
        exp >>= 1;
    }
    return r;
}

__device__ void d_horner_step(gl64_t *cumul, uint32_t num_challenges, gl64_t term, const uint64_t *alphas_u64)
{
    for (uint32_t ch = 0; ch < num_challenges; ++ch) {
        gl64_t al = gl64_t(alphas_u64[ch]);
        cumul[ch] = term + al * cumul[ch];
    }
}

/** Batch inversion in place (Montgomery trick), n >= 1. */
__device__ void d_batch_inv_in_place(gl64_t *a, int n)
{
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        a[0] = gl64_t::one() / a[0];
        return;
    }
    gl64_t prod[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    prod[0] = a[0];
    for (int i = 1; i < n; ++i) {
        prod[i] = prod[i - 1] * a[i];
    }
    gl64_t inv_all = gl64_t::one() / prod[n - 1];
    for (int i = n - 1; i > 0; --i) {
        gl64_t tmp = inv_all * prod[i - 1];
        inv_all = inv_all * a[i];
        a[i] = tmp;
    }
    a[0] = inv_all;
}

/* Device compilation pass has no host kernel launches; suppress "never referenced" for __global__. */
#ifdef __CUDA_ARCH__
#pragma nv_diag_suppress 177
#endif

__global__ void gather_quotient_domain_kernel(
    const uint64_t *cs_lde,
    size_t cs_leaf,
    const uint64_t *wires_lde,
    size_t wires_leaf,
    const uint64_t *zp_lde,
    size_t zp_leaf,
    uint32_t deg_log,
    uint32_t rate_bits,
    size_t step,
    size_t lde_q,
    uint32_t next_step,
    uint32_t num_constants,
    uint32_t num_routed,
    uint32_t num_wires,
    uint32_t zp_num_polys,
    uint32_t num_challenges,
    uint64_t *out_const,
    uint64_t *out_sigma,
    uint64_t *out_wires,
    uint64_t *out_zp,
    uint64_t *out_z_next)
{
    size_t pt = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (pt >= lde_q) {
        return;
    }
    size_t tb = (size_t)deg_log + (size_t)rate_bits;
    size_t raw = pt * step;
    size_t rev = d_rev_bits(raw, tb);
    size_t cs_off = rev * cs_leaf;
    size_t w_off = rev * wires_leaf;
    size_t z_off = rev * zp_leaf;

    size_t pt_n = (pt + (size_t)next_step) % lde_q;
    size_t raw_n = pt_n * step;
    size_t rev_n = d_rev_bits(raw_n, tb);
    size_t z_off_n = rev_n * zp_leaf;

    for (uint32_t c = 0; c < num_constants; ++c) {
        out_const[pt * (size_t)num_constants + c] = cs_lde[cs_off + c];
    }
    for (uint32_t s = 0; s < num_routed; ++s) {
        out_sigma[pt * (size_t)num_routed + s] = cs_lde[cs_off + (size_t)num_constants + s];
    }
    for (uint32_t w = 0; w < num_wires; ++w) {
        out_wires[pt * (size_t)num_wires + w] = wires_lde[w_off + w];
    }
    for (uint32_t z = 0; z < zp_num_polys; ++z) {
        out_zp[pt * (size_t)zp_num_polys + z] = zp_lde[z_off + z];
    }
    for (uint32_t ch = 0; ch < num_challenges; ++ch) {
        out_z_next[pt * (size_t)num_challenges + ch] = zp_lde[z_off_n + ch];
    }
}

/**
 * One thread per quotient LDE point. Writes `num_challenges` reduced values multiplied by Z_H^{-1}
 * (Plonky2 periodic table).
 */
__global__ void quotient_vanishing_reduce_kernel(
    size_t lde_q,
    uint32_t num_challenges,
    uint32_t num_pp_polys,
    uint32_t zp_np,
    uint32_t quotient_degree_factor,
    uint32_t num_routed,
    uint32_t num_wires,
    uint32_t num_gate_constraints,
    uint32_t quotient_degree_bits,
    uint64_t omega_quotient_u64,
    uint64_t coset_shift_u64,
    const uint64_t *d_const_pt,
    const uint64_t *d_wires_pt,
    const uint64_t *d_sigma_pt,
    const uint64_t *d_zp_pt,
    const uint64_t *d_z_next_pt,
    const uint64_t *d_gate,
    const uint64_t *d_k_is,
    const uint64_t *d_betas,
    const uint64_t *d_gammas,
    const uint64_t *d_alphas,
    const uint64_t *d_zh_inv_period,
    const uint64_t *d_zh_eval_period,
    uint64_t n_trace_u64,
    uint64_t *d_out)
{
    size_t pt = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (pt >= lde_q) {
        return;
    }

    if (num_routed > PARTIAL_PRODUCTS_MAX_ROUTED_WIRES) {
        return;
    }

    gl64_t omega = gl64_t(omega_quotient_u64);
    gl64_t g = gl64_t(coset_shift_u64);
    gl64_t x = g * d_gl64_pow_u64(omega, (uint64_t)pt);
    gl64_t n_field = gl64_t(n_trace_u64);

    uint32_t period = 1u << quotient_degree_bits;
    size_t pid = pt & (size_t)(period - 1u);
    gl64_t zh_tab = gl64_t(d_zh_eval_period[pid]);
    gl64_t l_0;
    {
        gl64_t den = n_field * (x - gl64_t::one());
        l_0 = zh_tab / den;
    }

    gl64_t cumul[32];
    if (num_challenges > 32) {
        return;
    }
    for (uint32_t ch = 0; ch < num_challenges; ++ch) {
        cumul[ch] = gl64_t::zero();
    }

    /* Reverse Horner: gate constraints last -> first. */
    for (int gc = (int)num_gate_constraints - 1; gc >= 0; --gc) {
        gl64_t term = gl64_t(d_gate[(size_t)gc * lde_q + pt]);
        d_horner_step(cumul, num_challenges, term, d_alphas);
    }

    uint32_t qdf = quotient_degree_factor;

    gl64_t numer[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];
    gl64_t denom[PARTIAL_PRODUCTS_MAX_ROUTED_WIRES];

    for (int chi = (int)num_challenges - 1; chi >= 0; --chi) {
        uint32_t ch = (uint32_t)chi;
        gl64_t beta = gl64_t(d_betas[ch]);
        gl64_t gamma = gl64_t(d_gammas[ch]);

        for (uint32_t j = 0; j < num_routed; ++j) {
            gl64_t wv = gl64_t(d_wires_pt[pt * (size_t)num_wires + j]);
            gl64_t k_i = gl64_t(d_k_is[j]);
            numer[j] = wv + beta * (k_i * x) + gamma;
            gl64_t sig = gl64_t(d_sigma_pt[pt * (size_t)num_routed + j]);
            denom[j] = wv + beta * sig + gamma;
        }
        d_batch_inv_in_place(denom, (int)num_routed);

        size_t zbase = pt * (size_t)zp_np;
        gl64_t z_x = gl64_t(d_zp_pt[zbase + (size_t)ch]);
        gl64_t z_gx = gl64_t(d_z_next_pt[pt * (size_t)num_challenges + (size_t)ch]);
        size_t pp_base = zbase + (size_t)num_challenges + (size_t)ch * (size_t)num_pp_polys;

        int num_checks = (int)num_pp_polys + 1;
        for (int cidx = num_checks - 1; cidx >= 0; --cidx) {
            size_t c = (size_t)cidx;
            size_t start = c * (size_t)qdf;
            size_t end = start + (size_t)qdf;
            if (end > (size_t)num_routed) {
                end = (size_t)num_routed;
            }
            gl64_t nprod = gl64_t::one();
            gl64_t dprod = gl64_t::one();
            for (size_t j = start; j < end; ++j) {
                nprod = nprod * numer[j];
                dprod = dprod * denom[j];
            }
            gl64_t prev_acc = (c == 0) ? z_x : gl64_t(d_zp_pt[pp_base + (c - 1)]);
            gl64_t next_acc =
                (c == (size_t)num_pp_polys) ? z_gx : gl64_t(d_zp_pt[pp_base + c]);
            gl64_t term = prev_acc * nprod - next_acc * dprod;
            d_horner_step(cumul, num_challenges, term, d_alphas);
        }

        gl64_t z_term = l_0 * (z_x - gl64_t::one());
        d_horner_step(cumul, num_challenges, z_term, d_alphas);
    }

    gl64_t zh_inv = gl64_t(d_zh_inv_period[pid]);
    for (uint32_t ch = 0; ch < num_challenges; ++ch) {
        d_out[(size_t)ch * lde_q + pt] = (uint64_t)(cumul[ch] * zh_inv);
    }
}

#ifdef __CUDA_ARCH__
#pragma nv_diag_default 177
#endif

} // namespace

#ifndef __CUDA_ARCH__

namespace {

struct cuda_u64_deleter {
    void operator()(uint64_t *p) const noexcept
    {
        if (p) {
            (void)cudaFree((void *)p);
        }
    }
};

using unique_u64 = std::unique_ptr<uint64_t, cuda_u64_deleter>;

struct QuotientWorkspace {
    uint32_t cached_deg_log = UINT32_MAX;
    uint32_t cached_q_bits = UINT32_MAX;

    size_t zh_eval_count = 0;
    size_t zh_inv_count = 0;
    size_t const_count = 0;
    size_t sigma_count = 0;
    size_t wires_count = 0;
    size_t zp_count = 0;
    size_t zn_count = 0;
    size_t gate_count = 0;
    size_t k_count = 0;
    size_t beta_count = 0;
    size_t gamma_count = 0;
    size_t alpha_count = 0;
    size_t qvals_count = 0;

    unique_u64 d_zh_eval{nullptr};
    unique_u64 d_zh_inv{nullptr};
    unique_u64 d_const{nullptr};
    unique_u64 d_sigma{nullptr};
    unique_u64 d_wires{nullptr};
    unique_u64 d_zp{nullptr};
    unique_u64 d_zn{nullptr};
    unique_u64 d_gate{nullptr};
    unique_u64 d_k{nullptr};
    unique_u64 d_beta{nullptr};
    unique_u64 d_gamma{nullptr};
    unique_u64 d_alpha{nullptr};
    unique_u64 d_qvals{nullptr};
};

std::mutex g_workspace_mu;
std::unordered_map<size_t, QuotientWorkspace> g_workspace_by_gpu;

inline void ensure_u64_buffer(unique_u64 &buf, size_t &capacity, size_t need)
{
    if (capacity >= need) {
        return;
    }
    uint64_t *ptr = nullptr;
    CUDA_OK(cudaMalloc(&ptr, need * sizeof(uint64_t)));
    buf.reset(ptr);
    capacity = need;
}

} // namespace

static inline uint32_t host_log2_ceil_u32(uint32_t n)
{
    if (n <= 1) {
        return 0;
    }
    return 32u - (uint32_t)__builtin_clz(n - 1);
}

static cpp_gl64_t cpp_pow_u64(cpp_gl64_t base, uint64_t exp)
{
    cpp_gl64_t r = cpp_gl64_t::one();
    cpp_gl64_t b = base;
    while (exp > 0) {
        if (exp & 1ULL) {
            r = r * b;
        }
        b = b * b;
        exp >>= 1;
    }
    return r;
}

/** Match `plonky2::field::zero_poly_coset::ZeroPolyOnCoset::new(degree_bits, quotient_degree_bits)`. */
static void build_zero_poly_on_coset_tables(
    uint32_t degree_bits,
    uint32_t quotient_degree_bits,
    std::vector<uint64_t> *eval_out,
    std::vector<uint64_t> *inv_out)
{
    uint32_t rate = 1u << quotient_degree_bits;
    eval_out->resize(rate);
    inv_out->resize(rate);
    cpp_gl64_t g = cpp_gl64_t(GROUP_GENERATOR);
    cpp_gl64_t g_pow_n = cpp_pow_u64(g, 1ULL << degree_bits);
    cpp_gl64_t omega_q = cpp_gl64_t(OMEGA[quotient_degree_bits]);
    std::vector<cpp_gl64_t> evals(rate);
    for (uint32_t j = 0; j < rate; ++j) {
        cpp_gl64_t wj = cpp_pow_u64(omega_q, (uint64_t)j);
        evals[j] = g_pow_n * wj - cpp_gl64_t::one();
        (*eval_out)[j] = evals[j].get_val();
    }
    for (uint32_t j = 0; j < rate; ++j) {
        if (evals[j].get_val() == 0) {
            (*inv_out)[j] = 0;
        } else {
            (*inv_out)[j] = inv_base(evals[j]).get_val();
        }
    }
}

RustError compute_quotient_polys_lde_pointers_gl64(
    size_t gpu_id,
    cudaStream_t stream,
    const uint64_t *d_cs_lde,
    uint64_t cs_leaf,
    const uint64_t *d_wires_lde,
    uint64_t wires_leaf,
    const uint64_t *d_zp_lde,
    uint64_t zp_leaf,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const uint64_t *h_k_is,
    const uint64_t *h_public_inputs_hash,
    const uint64_t *h_betas,
    const uint64_t *h_gammas,
    const uint64_t *h_alphas,
    fr_t *d_out_quotient_coeffs,
    size_t *out_lde_q_size)
{
    (void)h_public_inputs_hash;
    if (!d_cs_lde || !d_wires_lde || !d_zp_lde) {
        return RustError{EINVAL, "compute_quotient_polys: null LDE pointer"};
    }
    if (!config || !out_lde_q_size || !d_out_quotient_coeffs) {
        return RustError{EINVAL, "compute_quotient_polys: null argument"};
    }
    if (config->quotient_degree_factor < 2) {
        return RustError{EINVAL, "compute_quotient_polys: quotient_degree_factor must be >= 2"};
    }
    if (config->num_challenges == 0 || config->num_challenges > 32) {
        return RustError{EINVAL, "compute_quotient_polys: num_challenges out of range (1..=32)"};
    }
    if (config->num_routed_wires > PARTIAL_PRODUCTS_MAX_ROUTED_WIRES) {
        return RustError{EINVAL, "compute_quotient_polys: num_routed_wires exceeds GPU limit"};
    }

    uint32_t q_bits = host_log2_ceil_u32(config->quotient_degree_factor);
    if (q_bits > config->rate_bits) {
        return RustError{EINVAL, "compute_quotient_polys: quotient degree bits exceed rate_bits"};
    }

    uint32_t deg_log = config->degree_bits;
    uint32_t rate_bits = config->rate_bits;
    size_t step = (size_t)1u << (rate_bits - q_bits);
    uint32_t next_step = 1u << q_bits;
    size_t lde_q = (size_t)1u << ((size_t)deg_log + q_bits);
    *out_lde_q_size = lde_q;

    uint32_t num_constants = config->num_constants;
    uint32_t num_routed = config->num_routed_wires;
    uint32_t num_wires = config->num_wires;
    uint32_t nc = config->num_challenges;
    uint32_t num_pp = config->num_partial_products;
    uint32_t zp_np = nc + nc * num_pp;

    if (cs_leaf < (uint64_t)num_constants + (uint64_t)num_routed) {
        return RustError{EINVAL, "compute_quotient_polys: constants_sigmas leaf too small"};
    }
    if (wires_leaf < (uint64_t)num_wires) {
        return RustError{EINVAL, "compute_quotient_polys: wires leaf too small"};
    }
    if (zp_leaf < (uint64_t)zp_np) {
        return RustError{EINVAL, "compute_quotient_polys: zs_partial_products leaf too small"};
    }

    auto &gpu = select_gpu(gpu_id);
    gpu.select();

    std::lock_guard<std::mutex> lock(g_workspace_mu);
    QuotientWorkspace &ws = g_workspace_by_gpu[gpu_id];

    if (ws.cached_deg_log != deg_log || ws.cached_q_bits != q_bits) {
        std::vector<uint64_t> zh_eval_host;
        std::vector<uint64_t> zh_inv_host;
        build_zero_poly_on_coset_tables(deg_log, q_bits, &zh_eval_host, &zh_inv_host);

        ensure_u64_buffer(ws.d_zh_eval, ws.zh_eval_count, zh_eval_host.size());
        ensure_u64_buffer(ws.d_zh_inv, ws.zh_inv_count, zh_inv_host.size());
        CUDA_OK(cudaMemcpyAsync(
            ws.d_zh_eval.get(),
            zh_eval_host.data(),
            zh_eval_host.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_OK(cudaMemcpyAsync(
            ws.d_zh_inv.get(),
            zh_inv_host.data(),
            zh_inv_host.size() * sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            stream));

        ws.cached_deg_log = deg_log;
        ws.cached_q_bits = q_bits;
    }

    ensure_u64_buffer(ws.d_const, ws.const_count, lde_q * (size_t)num_constants);
    ensure_u64_buffer(ws.d_sigma, ws.sigma_count, lde_q * (size_t)num_routed);
    ensure_u64_buffer(ws.d_wires, ws.wires_count, lde_q * (size_t)num_wires);
    ensure_u64_buffer(ws.d_zp, ws.zp_count, lde_q * (size_t)zp_np);
    ensure_u64_buffer(ws.d_zn, ws.zn_count, lde_q * (size_t)nc);
    ensure_u64_buffer(ws.d_gate, ws.gate_count, (size_t)config->num_gate_constraints * lde_q);
    ensure_u64_buffer(ws.d_k, ws.k_count, (size_t)num_routed);
    ensure_u64_buffer(ws.d_beta, ws.beta_count, (size_t)nc);
    ensure_u64_buffer(ws.d_gamma, ws.gamma_count, (size_t)nc);
    ensure_u64_buffer(ws.d_alpha, ws.alpha_count, (size_t)nc);
    ensure_u64_buffer(ws.d_qvals, ws.qvals_count, (size_t)nc * lde_q);

    CUDA_OK(
        cudaMemcpyAsync(ws.d_k.get(), h_k_is, (size_t)num_routed * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(
        cudaMemcpyAsync(ws.d_beta.get(), h_betas, (size_t)nc * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(
        cudaMemcpyAsync(ws.d_gamma.get(), h_gammas, (size_t)nc * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_OK(
        cudaMemcpyAsync(ws.d_alpha.get(), h_alphas, (size_t)nc * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    int gather_threads = 256;
    int gather_blocks = (int)((lde_q + (size_t)gather_threads - 1) / (size_t)gather_threads);
    gather_quotient_domain_kernel<<<gather_blocks, gather_threads, 0, stream>>>(
        d_cs_lde,
        (size_t)cs_leaf,
        d_wires_lde,
        (size_t)wires_leaf,
        d_zp_lde,
        (size_t)zp_leaf,
        deg_log,
        rate_bits,
        step,
        lde_q,
        next_step,
        num_constants,
        num_routed,
        num_wires,
        zp_np,
        nc,
        ws.d_const.get(),
        ws.d_sigma.get(),
        ws.d_wires.get(),
        ws.d_zp.get(),
        ws.d_zn.get());
    CUDA_OK(cudaGetLastError());

    CUDA_OK(cudaMemsetAsync(
        ws.d_gate.get(), 0, (size_t)config->num_gate_constraints * lde_q * sizeof(uint64_t), stream));

    size_t n_gc = (size_t)config->num_gate_constraints;
    size_t row = 0;
    for (size_t gi = 0; gi < (size_t)num_gates && row < n_gc; ++gi) {
        const GateInfo &g = gates[gi];
        if (g.gate_type == (uint32_t)GateType::ArithmeticGate) {
            uint32_t nops = g.aux_0 > 0 ? g.aux_0 : 1u;
            for (uint32_t op = 0; op < nops && row < n_gc; ++op, ++row) {
                launch_eval_arithmetic_gate_constraints(
                    ws.d_const.get(),
                    ws.d_wires.get(),
                    lde_q,
                    (size_t)num_constants,
                    (size_t)num_wires,
                    g.wire_0 + 4u * op,
                    g.wire_1 + 4u * op,
                    g.wire_2 + 4u * op,
                    g.wire_3 + 4u * op,
                    g.const_0,
                    g.const_1,
                    row,
                    ws.d_gate.get(),
                    n_gc,
                    stream);
            }
        } else if (g.gate_type == (uint32_t)GateType::ConstantGate) {
            uint32_t nconst = g.aux_0 > 0 ? g.aux_0 : 1u;
            for (uint32_t j = 0; j < nconst && row < n_gc; ++j, ++row) {
                launch_eval_constant_gate_constraints(
                    ws.d_const.get(),
                    ws.d_wires.get(),
                    lde_q,
                    (size_t)num_constants,
                    (size_t)num_wires,
                    g.wire_0 + j,
                    g.const_0 + j,
                    row,
                    ws.d_gate.get(),
                    n_gc,
                    stream);
            }
        }
    }

    uint32_t lde_q_log = deg_log + q_bits;
    if (lde_q_log >= sizeof(OMEGA) / sizeof(OMEGA[0])) {
        return RustError{EINVAL, "compute_quotient_polys: quotient LDE log too large"};
    }

    int van_threads = 256;
    int van_blocks = (int)((lde_q + (size_t)van_threads - 1) / (size_t)van_threads);
    quotient_vanishing_reduce_kernel<<<van_blocks, van_threads, 0, stream>>>(
        lde_q,
        nc,
        num_pp,
        zp_np,
        config->quotient_degree_factor,
        num_routed,
        num_wires,
        config->num_gate_constraints,
        q_bits,
        OMEGA[lde_q_log],
        GROUP_GENERATOR,
        ws.d_const.get(),
        ws.d_wires.get(),
        ws.d_sigma.get(),
        ws.d_zp.get(),
        ws.d_zn.get(),
        ws.d_gate.get(),
        ws.d_k.get(),
        ws.d_beta.get(),
        ws.d_gamma.get(),
        ws.d_alpha.get(),
        ws.d_zh_inv.get(),
        ws.d_zh_eval.get(),
        1ULL << deg_log,
        ws.d_qvals.get());
    CUDA_OK(cudaGetLastError());

    NTT_Config inv_cfg = {};
    inv_cfg.batches = nc;
    inv_cfg.order = NN;
    inv_cfg.ntt_type = standard;
    inv_cfg.extension_rate_bits = 0;
    inv_cfg.are_inputs_on_device = true;
    inv_cfg.are_outputs_on_device = true;
    inv_cfg.with_coset = true;
    inv_cfg.is_multi_gpu = false;
    inv_cfg.salt_size = 0;

    RustError ierr =
        ntt::batch_ntt(gpu, reinterpret_cast<fr_t *>(ws.d_qvals.get()), (uint32_t)lde_q_log, inverse, inv_cfg);
    if (ierr.code != 0) {
        return ierr;
    }

    CUDA_OK(cudaMemcpyAsync(
        d_out_quotient_coeffs,
        ws.d_qvals.get(),
        (size_t)nc * lde_q * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return RustError{0};
}

RustError compute_quotient_polys_from_batches_gl64(
    size_t gpu_id,
    cudaStream_t stream,
    const PolynomialBatchGPU &constants_sigmas,
    const PolynomialBatchGPU &wires,
    const PolynomialBatchGPU &zs_partial_products,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const uint64_t *h_k_is,
    const uint64_t *h_public_inputs_hash,
    const uint64_t *h_betas,
    const uint64_t *h_gammas,
    const uint64_t *h_alphas,
    fr_t *d_out_quotient_coeffs,
    size_t *out_lde_q_size)
{
    return compute_quotient_polys_lde_pointers_gl64(
        gpu_id,
        stream,
        reinterpret_cast<const uint64_t *>(constants_sigmas.lde_gpu),
        constants_sigmas.leaf_size,
        reinterpret_cast<const uint64_t *>(wires.lde_gpu),
        wires.leaf_size,
        reinterpret_cast<const uint64_t *>(zs_partial_products.lde_gpu),
        zs_partial_products.leaf_size,
        config,
        gates,
        num_gates,
        h_k_is,
        h_public_inputs_hash,
        h_betas,
        h_gammas,
        h_alphas,
        d_out_quotient_coeffs,
        out_lde_q_size);
}

extern "C" RustError zeknox_compute_quotient_polys_gl64(
    size_t gpu_id,
    void *cuda_stream,
    const uint64_t *d_constants_sigmas_lde,
    uint64_t constants_sigmas_leaf_size,
    const uint64_t *d_wires_lde,
    uint64_t wires_leaf_size,
    const uint64_t *d_zs_partial_products_lde,
    uint64_t zs_partial_leaf_size,
    const ProverConfig *config,
    const GateInfo *gates,
    uint32_t num_gates,
    const uint64_t *h_k_is,
    const uint64_t *h_public_inputs_hash,
    const uint64_t *h_betas,
    const uint64_t *h_gammas,
    const uint64_t *h_alphas,
    uint64_t *d_out_quotient_coeffs,
    size_t *out_lde_q_size)
{
    return compute_quotient_polys_lde_pointers_gl64(
        gpu_id,
        static_cast<cudaStream_t>(cuda_stream),
        d_constants_sigmas_lde,
        constants_sigmas_leaf_size,
        d_wires_lde,
        wires_leaf_size,
        d_zs_partial_products_lde,
        zs_partial_leaf_size,
        config,
        gates,
        num_gates,
        h_k_is,
        h_public_inputs_hash,
        h_betas,
        h_gammas,
        h_alphas,
        reinterpret_cast<fr_t *>(d_out_quotient_coeffs),
        out_lde_q_size);
}

#endif // !__CUDA_ARCH__
