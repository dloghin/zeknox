// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include "gate_constraints.cuh"

#ifdef USE_CUDA

#include "types/int_types.h"
#include "ff/goldilocks.hpp"

static constexpr int GATE_KERNEL_BLOCK = 256;

__global__ void eval_arithmetic_gate_constraints(
    const uint64_t *local_constants,
    const uint64_t *local_wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_a,
    uint32_t wire_b,
    uint32_t wire_c,
    uint32_t wire_out,
    uint32_t const_c0,
    uint32_t const_c1,
    size_t constraint_row,
    uint64_t *constraint_accumulator,
    size_t num_gate_constraint_rows)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= num_points || constraint_row >= num_gate_constraint_rows) {
        return;
    }

    gl64_t a = gl64_t(local_wires[i * num_wires + wire_a]);
    gl64_t b = gl64_t(local_wires[i * num_wires + wire_b]);
    gl64_t c = gl64_t(local_wires[i * num_wires + wire_c]);
    gl64_t out = gl64_t(local_wires[i * num_wires + wire_out]);
    gl64_t k0 = gl64_t(local_constants[i * num_constants + const_c0]);
    gl64_t k1 = gl64_t(local_constants[i * num_constants + const_c1]);

    gl64_t v = a * b * k0 + c * k1 - out;
    constraint_accumulator[constraint_row * num_points + i] = (uint64_t)v;
}

__global__ void eval_constant_gate_constraints(
    const uint64_t *local_constants,
    const uint64_t *local_wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_idx,
    uint32_t const_idx,
    size_t constraint_row,
    uint64_t *constraint_accumulator,
    size_t num_gate_constraint_rows)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= num_points || constraint_row >= num_gate_constraint_rows) {
        return;
    }

    gl64_t w = gl64_t(local_wires[i * num_wires + wire_idx]);
    gl64_t k = gl64_t(local_constants[i * num_constants + const_idx]);
    gl64_t v = w - k;
    constraint_accumulator[constraint_row * num_points + i] = (uint64_t)v;
}

void launch_eval_arithmetic_gate_constraints(
    const uint64_t *d_constants,
    const uint64_t *d_wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_a,
    uint32_t wire_b,
    uint32_t wire_c,
    uint32_t wire_out,
    uint32_t const_c0,
    uint32_t const_c1,
    size_t constraint_row,
    uint64_t *d_constraint_accumulator,
    size_t num_gate_constraint_rows,
    cudaStream_t stream)
{
    if (num_points == 0) {
        return;
    }
    int blocks = (int)((num_points + (size_t)GATE_KERNEL_BLOCK - 1) / (size_t)GATE_KERNEL_BLOCK);
    eval_arithmetic_gate_constraints<<<blocks, GATE_KERNEL_BLOCK, 0, stream>>>(
        d_constants,
        d_wires,
        num_points,
        num_constants,
        num_wires,
        wire_a,
        wire_b,
        wire_c,
        wire_out,
        const_c0,
        const_c1,
        constraint_row,
        d_constraint_accumulator,
        num_gate_constraint_rows);
}

void launch_eval_constant_gate_constraints(
    const uint64_t *d_constants,
    const uint64_t *d_wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_idx,
    uint32_t const_idx,
    size_t constraint_row,
    uint64_t *d_constraint_accumulator,
    size_t num_gate_constraint_rows,
    cudaStream_t stream)
{
    if (num_points == 0) {
        return;
    }
    int blocks = (int)((num_points + (size_t)GATE_KERNEL_BLOCK - 1) / (size_t)GATE_KERNEL_BLOCK);
    eval_constant_gate_constraints<<<blocks, GATE_KERNEL_BLOCK, 0, stream>>>(
        d_constants,
        d_wires,
        num_points,
        num_constants,
        num_wires,
        wire_idx,
        const_idx,
        constraint_row,
        d_constraint_accumulator,
        num_gate_constraint_rows);
}

#ifndef __CUDA_ARCH__

static uint64_t canon_gl64_val(uint64_t v)
{
    if (v == cpp_gl64_t::MOD) {
        return 0;
    }
    return v;
}

void gate_constraints_arithmetic_cpu_reference(
    const uint64_t *constants,
    const uint64_t *wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_a,
    uint32_t wire_b,
    uint32_t wire_c,
    uint32_t wire_out,
    uint32_t const_c0,
    uint32_t const_c1,
    uint64_t *out)
{
    for (size_t i = 0; i < num_points; ++i) {
        cpp_gl64_t a = cpp_gl64_t(wires[i * num_wires + wire_a]);
        cpp_gl64_t b = cpp_gl64_t(wires[i * num_wires + wire_b]);
        cpp_gl64_t c = cpp_gl64_t(wires[i * num_wires + wire_c]);
        cpp_gl64_t wo = cpp_gl64_t(wires[i * num_wires + wire_out]);
        cpp_gl64_t k0 = cpp_gl64_t(constants[i * num_constants + const_c0]);
        cpp_gl64_t k1 = cpp_gl64_t(constants[i * num_constants + const_c1]);
        cpp_gl64_t v = a * b * k0 + c * k1 - wo;
        out[i] = canon_gl64_val(v.get_val());
    }
}

void gate_constraints_constant_cpu_reference(
    const uint64_t *constants,
    const uint64_t *wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_idx,
    uint32_t const_idx,
    uint64_t *out)
{
    for (size_t i = 0; i < num_points; ++i) {
        cpp_gl64_t w = cpp_gl64_t(wires[i * num_wires + wire_idx]);
        cpp_gl64_t k = cpp_gl64_t(constants[i * num_constants + const_idx]);
        cpp_gl64_t v = w - k;
        out[i] = canon_gl64_val(v.get_val());
    }
}

#endif // !__CUDA_ARCH__

#endif // USE_CUDA
