// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cerrno>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <ff/gl64_params.hpp>
#include <ff/goldilocks.hpp>
#include <merkle/hasher.hpp>
#include <prover/types.h>
#include <utils/rusterror.h>

extern "C" {
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
    uint64_t gpu_id);
}

static void cuda_check(cudaError_t e, const char *ctx)
{
    ASSERT_EQ(e, cudaSuccess) << ctx << ": " << cudaGetErrorString(e);
}

TEST(GpuProve, rejects_null_args)
{
    ProverConfig cfg = {};
    size_t sz = 0;
    RustError e = gpu_prove(
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &cfg,
        nullptr,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
        &sz,
        0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_invalid_config)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 32; // > 31, invalid
    cfg.num_wires = 4;
    cfg.num_routed_wires = 2;
    cfg.quotient_degree_factor = 2;
    size_t sz = 1024;
    uint8_t buf[16];
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        nullptr, 0, nullptr,
        buf, &sz, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_nonzero_fri_rounds)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 4;
    cfg.num_wires = 4;
    cfg.num_routed_wires = 2;
    cfg.quotient_degree_factor = 2;
    size_t sz = 1024;
    uint8_t buf[16];
    uint32_t arity = 1;
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        &arity, 1, nullptr,
        buf, &sz, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_null_proof_size)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 4;
    cfg.num_wires = 4;
    cfg.num_routed_wires = 2;
    cfg.quotient_degree_factor = 2;
    uint8_t buf[16];
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        nullptr, 0, nullptr,
        buf, nullptr, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_zero_num_wires)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 4;
    cfg.num_wires = 0;
    cfg.num_routed_wires = 2;
    cfg.quotient_degree_factor = 2;
    size_t sz = 1024;
    uint8_t buf[16];
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        nullptr, 0, nullptr,
        buf, &sz, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_zero_num_routed_wires)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 4;
    cfg.num_wires = 4;
    cfg.num_routed_wires = 0;
    cfg.quotient_degree_factor = 2;
    size_t sz = 1024;
    uint8_t buf[16];
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        nullptr, 0, nullptr,
        buf, &sz, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, rejects_zero_quotient_degree_factor)
{
    ProverConfig cfg = {};
    cfg.degree_bits = 4;
    cfg.num_wires = 4;
    cfg.num_routed_wires = 2;
    cfg.quotient_degree_factor = 0;
    size_t sz = 1024;
    uint8_t buf[16];
    RustError e = gpu_prove(
        (void *)1, nullptr, nullptr, nullptr,
        (void *)1, (void *)1, &cfg, nullptr, 0,
        (void *)1, (void *)1, (void *)1,
        nullptr, 0, nullptr,
        buf, &sz, 0);
    ASSERT_NE(e.code, 0);
}

TEST(GpuProve, orchestrator_smoke)
{
    const uint32_t LOG_DEGREE = 5;
    const uint32_t DEGREE = 1u << LOG_DEGREE;
    const uint32_t RATE_BITS = 2;
    const uint32_t CAP_HEIGHT = 1;
    const uint32_t NUM_WIRES = 4;
    const uint32_t NUM_ROUTED = 2;
    const uint32_t NUM_CONST = 2;
    const uint32_t QDF = 2;
    const uint32_t NUM_CS = NUM_CONST + NUM_ROUTED;

    std::vector<uint64_t> h_wire(NUM_WIRES * DEGREE);
    std::vector<uint64_t> h_cs_sig(NUM_CS * DEGREE);
    for (uint32_t i = 0; i < h_wire.size(); i++) {
        h_wire[i] = ((uint64_t)(i * 17 + 3)) % cpp_gl64_t::MOD;
    }
    for (uint32_t i = 0; i < h_cs_sig.size(); i++) {
        h_cs_sig[i] = ((uint64_t)(i * 31 + 5)) % cpp_gl64_t::MOD;
    }

    std::vector<uint64_t> h_sub(DEGREE);
    {
        const uint64_t omega = OMEGA[LOG_DEGREE];
        cpp_gl64_t w(omega);
        cpp_gl64_t acc = cpp_gl64_t::one();
        for (uint32_t i = 0; i < DEGREE; i++) {
            h_sub[i] = acc.get_val();
            acc = acc * w;
        }
    }
    std::vector<uint64_t> h_k(NUM_ROUTED);
    h_k[0] = 3;
    h_k[1] = 5;

    gl64_t digest[NUM_HASH_OUT_ELTS] = {
        gl64_t(0x1010101010101010ULL),
        gl64_t(0x2020202020202020ULL),
        gl64_t(0x3030303030303030ULL),
        gl64_t(0x4040404040404040ULL),
    };
    gl64_t pubh[NUM_HASH_OUT_ELTS] = {
        gl64_t(1),
        gl64_t(2),
        gl64_t(3),
        gl64_t(4),
    };

    fr_t *d_cs = nullptr;
    cuda_check(cudaMalloc(&d_cs, h_cs_sig.size() * sizeof(fr_t)), "cudaMalloc");
    cuda_check(
        cudaMemcpy(d_cs, h_cs_sig.data(), h_cs_sig.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
        "cudaMemcpy");

    ProverConfig cfg = {};
    cfg.degree_bits = LOG_DEGREE;
    cfg.num_wires = NUM_WIRES;
    cfg.num_routed_wires = NUM_ROUTED;
    cfg.num_challenges = 1;
    cfg.num_partial_products = 1;
    cfg.quotient_degree_factor = QDF;
    cfg.rate_bits = RATE_BITS;
    cfg.cap_height = CAP_HEIGHT;
    cfg.num_gate_constraints = 0;
    cfg.num_constants = NUM_CONST;
    cfg.num_public_inputs = 0;

    std::vector<uint8_t> tiny(16);
    size_t proof_cap = tiny.size();
    RustError e0 = gpu_prove(
        d_cs,
        nullptr,
        nullptr,
        nullptr,
        digest,
        pubh,
        &cfg,
        nullptr,
        0,
        h_k.data(),
        h_sub.data(),
        h_wire.data(),
        nullptr,
        0,
        nullptr,
        tiny.data(),
        &proof_cap,
        0);
    ASSERT_EQ(e0.code, E2BIG);
    ASSERT_GT(proof_cap, tiny.size());

    std::vector<uint8_t> proof(proof_cap);
    size_t proof_size = proof.size();
    RustError e1 = gpu_prove(
        d_cs,
        nullptr,
        nullptr,
        nullptr,
        digest,
        pubh,
        &cfg,
        nullptr,
        0,
        h_k.data(),
        h_sub.data(),
        h_wire.data(),
        nullptr,
        0,
        nullptr,
        proof.data(),
        &proof_size,
        0);
    ASSERT_EQ(e1.code, 0) << (e1.message ? e1.message : "");
    ASSERT_EQ(proof_size, proof_cap);

    // Validate proof header (magic, version)
    ASSERT_GE(proof_size, 12u);
    uint32_t magic = 0, version = 0, sections = 0;
    memcpy(&magic, proof.data(), sizeof(uint32_t));
    memcpy(&version, proof.data() + 4, sizeof(uint32_t));
    memcpy(&sections, proof.data() + 8, sizeof(uint32_t));
    ASSERT_EQ(magic, (uint32_t)0x584e4b5a);
    ASSERT_EQ(version, 1u);
    ASSERT_EQ(sections, 1u);

    // Parse and validate cap lengths and opening set from the proof blob
    const uint8_t *p = proof.data() + 12;
    uint64_t wire_cap_len = 0;
    memcpy(&wire_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(wire_cap_len, 0u);
    p += wire_cap_len * sizeof(uint64_t);

    uint64_t zp_cap_len = 0;
    memcpy(&zp_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(zp_cap_len, 0u);
    p += zp_cap_len * sizeof(uint64_t);

    uint64_t q_cap_len = 0;
    memcpy(&q_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(q_cap_len, 0u);
    p += q_cap_len * sizeof(uint64_t);

    uint64_t opening_count = 0;
    memcpy(&opening_count, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(opening_count, 0u);
    p += opening_count * sizeof(uint64_t);

    // Verify the parsed size matches the reported proof_size
    ASSERT_EQ((size_t)(p - proof.data()), proof_size);

    // Run a second time with the same inputs and verify deterministic output
    std::vector<uint8_t> proof2(proof_cap);
    size_t proof_size2 = proof2.size();
    RustError e2 = gpu_prove(
        d_cs,
        nullptr,
        nullptr,
        nullptr,
        digest,
        pubh,
        &cfg,
        nullptr,
        0,
        h_k.data(),
        h_sub.data(),
        h_wire.data(),
        nullptr,
        0,
        nullptr,
        proof2.data(),
        &proof_size2,
        0);
    ASSERT_EQ(e2.code, 0) << (e2.message ? e2.message : "");
    ASSERT_EQ(proof_size2, proof_size);
    ASSERT_EQ(proof, proof2) << "Proof output is not deterministic";

    cudaFree(d_cs);
}

TEST(GpuProve, orchestrator_with_gate_constraints)
{
    const uint32_t LOG_DEGREE = 5;
    const uint32_t DEGREE = 1u << LOG_DEGREE;
    const uint32_t RATE_BITS = 2;
    const uint32_t CAP_HEIGHT = 1;
    const uint32_t NUM_WIRES = 4;
    const uint32_t NUM_ROUTED = 2;
    const uint32_t NUM_CONST = 2;
    const uint32_t QDF = 2;
    const uint32_t NUM_CS = NUM_CONST + NUM_ROUTED;
    const uint32_t NUM_GATE_CONSTRAINTS = 1;

    std::vector<uint64_t> h_wire(NUM_WIRES * DEGREE);
    std::vector<uint64_t> h_cs_sig(NUM_CS * DEGREE);
    for (uint32_t i = 0; i < h_wire.size(); i++) {
        h_wire[i] = ((uint64_t)(i * 17 + 3)) % cpp_gl64_t::MOD;
    }
    for (uint32_t i = 0; i < h_cs_sig.size(); i++) {
        h_cs_sig[i] = ((uint64_t)(i * 31 + 5)) % cpp_gl64_t::MOD;
    }

    std::vector<uint64_t> h_sub(DEGREE);
    {
        const uint64_t omega = OMEGA[LOG_DEGREE];
        cpp_gl64_t w(omega);
        cpp_gl64_t acc = cpp_gl64_t::one();
        for (uint32_t i = 0; i < DEGREE; i++) {
            h_sub[i] = acc.get_val();
            acc = acc * w;
        }
    }
    std::vector<uint64_t> h_k(NUM_ROUTED);
    h_k[0] = 3;
    h_k[1] = 5;

    gl64_t digest[NUM_HASH_OUT_ELTS] = {
        gl64_t(0x1010101010101010ULL),
        gl64_t(0x2020202020202020ULL),
        gl64_t(0x3030303030303030ULL),
        gl64_t(0x4040404040404040ULL),
    };
    gl64_t pubh[NUM_HASH_OUT_ELTS] = {
        gl64_t(1),
        gl64_t(2),
        gl64_t(3),
        gl64_t(4),
    };

    fr_t *d_cs = nullptr;
    cuda_check(cudaMalloc(&d_cs, h_cs_sig.size() * sizeof(fr_t)), "cudaMalloc");
    cuda_check(
        cudaMemcpy(d_cs, h_cs_sig.data(), h_cs_sig.size() * sizeof(uint64_t), cudaMemcpyHostToDevice),
        "cudaMemcpy");

    GateInfo gate = {};
    gate.gate_type = 0; // ArithmeticGate
    gate.wire_0 = 0;
    gate.wire_1 = 1;
    gate.wire_2 = 2;
    gate.wire_3 = 3;
    gate.const_0 = 0;
    gate.const_1 = 1;
    gate.aux_0 = 1;

    ProverConfig cfg = {};
    cfg.degree_bits = LOG_DEGREE;
    cfg.num_wires = NUM_WIRES;
    cfg.num_routed_wires = NUM_ROUTED;
    cfg.num_challenges = 1;
    cfg.num_partial_products = 1;
    cfg.quotient_degree_factor = QDF;
    cfg.rate_bits = RATE_BITS;
    cfg.cap_height = CAP_HEIGHT;
    cfg.num_gate_constraints = NUM_GATE_CONSTRAINTS;
    cfg.num_constants = NUM_CONST;
    cfg.num_public_inputs = 0;

    // First call: get required proof buffer size
    std::vector<uint8_t> tiny(16);
    size_t proof_cap = tiny.size();
    RustError e0 = gpu_prove(
        d_cs,
        nullptr, nullptr, nullptr,
        digest, pubh, &cfg,
        &gate, 1,
        h_k.data(), h_sub.data(), h_wire.data(),
        nullptr, 0, nullptr,
        tiny.data(), &proof_cap, 0);
    ASSERT_EQ(e0.code, E2BIG);
    ASSERT_GT(proof_cap, tiny.size());

    // Second call: produce the proof
    std::vector<uint8_t> proof(proof_cap);
    size_t proof_size = proof.size();
    RustError e1 = gpu_prove(
        d_cs,
        nullptr, nullptr, nullptr,
        digest, pubh, &cfg,
        &gate, 1,
        h_k.data(), h_sub.data(), h_wire.data(),
        nullptr, 0, nullptr,
        proof.data(), &proof_size, 0);
    ASSERT_EQ(e1.code, 0) << (e1.message ? e1.message : "");
    ASSERT_EQ(proof_size, proof_cap);

    // Validate proof header
    ASSERT_GE(proof_size, 12u);
    uint32_t magic = 0, version = 0, sections = 0;
    memcpy(&magic, proof.data(), sizeof(uint32_t));
    memcpy(&version, proof.data() + 4, sizeof(uint32_t));
    memcpy(&sections, proof.data() + 8, sizeof(uint32_t));
    ASSERT_EQ(magic, (uint32_t)0x584e4b5a);
    ASSERT_EQ(version, 1u);
    ASSERT_EQ(sections, 1u);

    // Parse proof blob: caps + openings
    const uint8_t *p = proof.data() + 12;
    uint64_t wire_cap_len = 0;
    memcpy(&wire_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(wire_cap_len, 0u);
    p += wire_cap_len * sizeof(uint64_t);

    uint64_t zp_cap_len = 0;
    memcpy(&zp_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(zp_cap_len, 0u);
    p += zp_cap_len * sizeof(uint64_t);

    uint64_t q_cap_len = 0;
    memcpy(&q_cap_len, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(q_cap_len, 0u);
    p += q_cap_len * sizeof(uint64_t);

    uint64_t opening_count = 0;
    memcpy(&opening_count, p, sizeof(uint64_t));
    p += sizeof(uint64_t);
    ASSERT_GT(opening_count, 0u);
    p += opening_count * sizeof(uint64_t);

    ASSERT_EQ((size_t)(p - proof.data()), proof_size);

    // Deterministic: same inputs produce same proof
    std::vector<uint8_t> proof2(proof_cap);
    size_t proof_size2 = proof2.size();
    RustError e2 = gpu_prove(
        d_cs,
        nullptr, nullptr, nullptr,
        digest, pubh, &cfg,
        &gate, 1,
        h_k.data(), h_sub.data(), h_wire.data(),
        nullptr, 0, nullptr,
        proof2.data(), &proof_size2, 0);
    ASSERT_EQ(e2.code, 0) << (e2.message ? e2.message : "");
    ASSERT_EQ(proof_size2, proof_size);
    ASSERT_EQ(proof, proof2) << "Proof with gate constraints is not deterministic";

    // The proof with gate constraints should differ from a proof without them
    // (since gate constraints produce non-zero quotient polynomial)
    ProverConfig cfg_no_gates = cfg;
    cfg_no_gates.num_gate_constraints = 0;

    std::vector<uint8_t> proof_no_gates(proof_cap * 2);
    size_t proof_size_ng = proof_no_gates.size();
    RustError e3 = gpu_prove(
        d_cs,
        nullptr, nullptr, nullptr,
        digest, pubh, &cfg_no_gates,
        nullptr, 0,
        h_k.data(), h_sub.data(), h_wire.data(),
        nullptr, 0, nullptr,
        proof_no_gates.data(), &proof_size_ng, 0);
    ASSERT_EQ(e3.code, 0) << (e3.message ? e3.message : "");
    proof_no_gates.resize(proof_size_ng);

    // Proofs should differ because gate constraints produce non-zero quotient polynomial values
    proof_no_gates.resize(proof_size_ng);
    bool proofs_differ = (proof_size != proof_size_ng) ||
                         (memcmp(proof.data(), proof_no_gates.data(),
                                 std::min(proof_size, proof_size_ng)) != 0);
    ASSERT_TRUE(proofs_differ)
        << "Proof with/without gates should differ (quotient polys change opening values)";

    cudaFree(d_cs);
}
