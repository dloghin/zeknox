// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_GATE_CONSTRAINTS_CUH__
#define __ZEKNOX_PROVER_GATE_CONSTRAINTS_CUH__

#include <cstddef>
#include <cstdint>

#ifdef USE_CUDA

/** Plonky2-style gate kinds (subset; extend as needed). */
enum class GateType : uint32_t {
    ArithmeticGate = 0,
    ConstantGate = 1,
    NoopGate = 2,
};

/**
 * Describes one gate instance in the constraint batch (offsets into per-point wire/constant arrays).
 * Layout matches a filtered batch: each gate reads a contiguous slice of wires/constants per point.
 */
struct GateDescriptor {
    uint32_t gate_type;
    uint32_t selector_index;
    uint32_t group_start;
    uint32_t group_end;
    uint32_t num_selectors;
};

/**
 * ArithmeticGate (base field): 4 wires (a,b,c,out) and 2 constants (c0,c1).
 * Constraint: a*b*c0 + c*c1 - out  (Plonky2 arithmetic gate shape).
 * Writes constraint value to constraint_accumulator[constraint_row * num_points + point].
 */
__global__ void eval_arithmetic_gate_constraints(
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
    size_t constraint_row,
    uint64_t *constraint_accumulator);

/**
 * ConstantGate: a wire must equal a constant (per point).
 * Constraint: `wire[wire_idx] - const[const_idx]`.
 */
__global__ void eval_constant_gate_constraints(
    const uint64_t *constants,
    const uint64_t *wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_idx,
    uint32_t const_idx,
    size_t constraint_row,
    uint64_t *constraint_accumulator);

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
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

/** Host references for tests (same formulas as kernels). */
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
    uint64_t *out);

void gate_constraints_constant_cpu_reference(
    const uint64_t *constants,
    const uint64_t *wires,
    size_t num_points,
    size_t num_constants,
    size_t num_wires,
    uint32_t wire_idx,
    uint32_t const_idx,
    uint64_t *out);

#endif // USE_CUDA

#endif // __ZEKNOX_PROVER_GATE_CONSTRAINTS_CUH__
