// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __ZEKNOX_PROVER_TYPES_H__
#define __ZEKNOX_PROVER_TYPES_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Configuration for `gpu_prove` (Plonky2-style trace / quotient parameters). */
typedef struct ProverConfig {
    uint32_t degree_bits;
    uint32_t num_wires;
    uint32_t num_routed_wires;
    uint32_t num_challenges;
    uint32_t num_partial_products;
    uint32_t quotient_degree_factor;
    uint32_t rate_bits;
    uint32_t cap_height;
    uint32_t num_gate_constraints;
    uint32_t num_constants;
    uint32_t num_public_inputs;
} ProverConfig;

/** One gate instance for future gate-constraint wiring (metadata only in the orchestrator MVP). */
typedef struct GateInfo {
    uint32_t gate_type;
    uint32_t selector_index;
    uint32_t group_start;
    uint32_t group_end;
    uint32_t num_selectors;
    uint32_t num_constraints;
    /* Optional wiring metadata for native quotient/gate-eval kernels. */
    uint32_t wire_0;
    uint32_t wire_1;
    uint32_t wire_2;
    uint32_t wire_3;
    uint32_t const_0;
    uint32_t const_1;
    uint32_t aux_0;
    uint32_t aux_1;
} GateInfo;

#ifdef __cplusplus
}
#endif

#endif // __ZEKNOX_PROVER_TYPES_H__
