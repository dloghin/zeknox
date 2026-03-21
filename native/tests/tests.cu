// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <math.h>

#include <keccak/keccak.hpp>
#include <monolith/monolith.hpp>
#include <poseidon/poseidon.hpp>
#include <poseidon2/poseidon2.hpp>
#include <poseidon/poseidon_bn128.hpp>
#include <merkle/merkle.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

/**
 * define DEBUG for printing
 */
// #define DEBUG

/**
 * define TIMING for printing execution time info
 */
// #define TIMING

#ifdef TIMING
#include <time.h>
#include <sys/time.h>
#endif // TIMING

#define HASH_SIZE_U64 4

#ifdef DEBUG
void printhash(u64 *h)
{
    for (int i = 0; i < 4; i++)
    {
        printf("%lu ", h[i]);
    }
    printf("\n");
}
#endif

#ifdef USE_CUDA

#include <utils/cuda_utils.cuh>

__global__ void keccak_gpu_driver(u64 *input, u32 size, u64 *hash)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    KeccakHasher::gpu_hash_one((gl64_t *)input, size, (gl64_t *)hash);
}

void keccak_hash_on_gpu(u64 *input, u32 size, u64 *hash)
{
    u64 *gpu_data, *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, size * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, input, size * sizeof(u64), cudaMemcpyHostToDevice));
    keccak_gpu_driver<<<1, 1>>>(gpu_data, size, gpu_hash);
    CHECKCUDAERR(cudaMemcpy(hash, gpu_hash, HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(gpu_data));
    CHECKCUDAERR(cudaFree(gpu_hash));
}

__global__ void monolith_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    MonolithHasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void monolith_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    MonolithHasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + HASH_SIZE_U64 * tid));
}

__global__ void monolith_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    MonolithHasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    PoseidonHasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    PoseidonHasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + HASH_SIZE_U64 * tid));
}

__global__ void poseidon_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    PoseidonHasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidon2_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    Poseidon2Hasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}

__global__ void poseidon2_hash_step1(u64 *in, u64 *out, u32 n, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    Poseidon2Hasher::gpu_hash_one((gl64_t *)(in + n * tid), n, (gl64_t *)(out + HASH_SIZE_U64 * tid));
}

__global__ void poseidon2_hash_step2(u64 *in, u64 *out, u32 len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len)
        return;

    Poseidon2Hasher::gpu_hash_two((gl64_t *)(in + 8 * tid), (gl64_t *)(in + 8 * tid + 4), (gl64_t *)(out + 4 * tid));
}

__global__ void poseidonbn128_hash(u64 *in, u64 *out, u32 n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    PoseidonBN128Hasher::gpu_hash_one((gl64_t *)in, n, (gl64_t *)out);
}
#endif

TEST(LIBCUDA, keccak_test)
{
    u64 data[6] = {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 13588825262671526271ul, 13421290117754017454ul, 7401888676587830362ul};

    u64 expected[7][HASH_SIZE_U64] = {
        {0ull},
        {13421290117754017454ul, 0, 0, 0ull},
        {13421290117754017454ul, 7401888676587830362ul, 0, 0ull},
        {13421290117754017454ul, 7401888676587830362ul, 15316685236050041751ul, 0ull},
        {9981707860959651334ul, 16351366398560378420ul, 4283762868800363615ul, 101ull},
        {708367124667950404ul, 17208681281141108820ul, 8334320481120086961ul, 134ull},
        {16109761546392287110ul, 4918745475135463511ul, 17110319063854316944ul, 103}};

    u64 h1[HASH_SIZE_U64] = {0u};
#ifdef USE_CUDA
    u64 h2[HASH_SIZE_U64] = {0u};
#endif

    for (int size = 1; size <= 6; size++)
    {
        KeccakHasher::cpu_hash_one(data, size, h1);
#ifdef USE_CUDA
        keccak_hash_on_gpu(data, size, h2);
#endif
#ifdef DEBUG
        printf("*** Size %d\n", size);
        printhash(h1);
        printhash(h2);
#endif
        for (int j = 0; j < HASH_SIZE_U64; j++)
        {
            assert(h1[j] == expected[size][j]);
#ifdef USE_CUDA
            assert(h2[j] == expected[size][j]);
#endif
        }
    }
}

TEST(LIBCUDA, monolith_test1)
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 expected[HASH_SIZE_U64] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

    u64 h1[HASH_SIZE_U64] = {0u};

    MonolithHasher::cpu_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif
    assert(h1[0] == expected[0]);
    assert(h1[1] == expected[1]);
    assert(h1[2] == expected[2]);
    assert(h1[3] == expected[3]);

#ifdef USE_CUDA
    u64 h2[HASH_SIZE_U64] = {0u};
    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash<<<1, 1>>>(gpu_data, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    printhash(h2);
#endif
    assert(h2[0] == expected[0]);
    assert(h2[1] == expected[1]);
    assert(h2[2] == expected[2]);
    assert(h2[3] == expected[3]);
#endif
}

#ifdef USE_CUDA
TEST(LIBCUDA, monolith_test2)
{
    // 4 leaves of 7 elements each -> Merkle tree has 7 nodes
    u64 test_leaves[28] = {
        12382199520291307008, 18193113598248284716, 17339479877015319223, 10837159358996869336, 9988531527727040483, 5682487500867411209, 13124187887292514366,
        8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027,
        10465118329878758468, 5866464242232862106, 15506463679657361352, 18404485636523119190, 15311871720566825080, 5967980567132965479, 14180845406393061616,
        15480539652174185186, 5454640537573844893, 3664852224809466446, 5547792914986991141, 5885254103823722535, 6014567676786509263, 11767239063322171808};

    // CPU
    u64 tree1[28] = {0ul};

    for (u32 i = 0; i < 4; i++)
    {
        MonolithHasher::cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    MonolithHasher::cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    MonolithHasher::cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    MonolithHasher::cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

    // GPU
    u64 tree2[28] = {0ul};

    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash_step1<<<1, 4>>>(gpu_data, gpu_hash, 7, 4);
    monolith_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    monolith_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, poseidon_test1)
{
    u64 leaf[9] = {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 3651132590097252162ull, 1141848076261006345ull, 12736915248278257710ull, 9898074228282442027ull, 16154511938222758243ull, 3651132590097252162ull};

    u64 expected[11][HASH_SIZE_U64] = {
        {0ull},
        {8395359103262935841ull, 0ull, 0ull, 0ull},
        {8395359103262935841ull, 1377884553022145855ull, 0ull, 0ull},
        {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 0ull},
        {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 3651132590097252162ull},
        {3618821072812614426ull, 8353148445756493727ull, 4040525329700581442ull, 15983474240847269257ull},
        {16643938361881363776ull, 6653675298471110559ull, 12562058402463703932ull, 16154511938222758243ull},
        {7544909477878586743ull, 7431000548126831493ull, 17815668806142634286ull, 13168106265494210017ull},
        {6835933650993053111ull, 15978194778874965616ull, 2024081381896137659ull, 16520693669262110264ull},
        {9429914239539731992ull, 14881719063945231827ull, 15528667124986963891ull, 16465743531992249573ull},
        {16643938361881363776ull, 6653675298471110559ull, 12562058402463703932ull, 16154511938222758243ull}};

    u64 h1[HASH_SIZE_U64] = {0u};

    for (int k = 1; k <= 9; k++)
    {
        PoseidonHasher::cpu_hash_one(leaf, k, h1);
#ifdef DEBUG
        printhash(h1);
#endif
        for (int j = 0; j < 4; j++)
        {
            assert(h1[j] == expected[k][j]);
        }
    }

#ifdef USE_CUDA
    u64 h2[HASH_SIZE_U64] = {0u};

    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 9 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, leaf, 9 * sizeof(u64), cudaMemcpyHostToDevice));

    for (int k = 1; k <= 9; k++)
    {
        poseidon_hash<<<1, 1>>>(gpu_leaf, gpu_hash, k);
        CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
        printhash(h2);
#endif // DEBUG
        for (int j = 0; j < HASH_SIZE_U64; j++)
        {
            assert(h2[j] == expected[k][j]);
        }
    }
#endif // USE_CUDA

#ifdef RUST_POSEIDON
    ext_poseidon_hash_or_noop(h1, leaf, 6);
    printhash(h1);
#endif // RUST_POSEIDON
}

#ifdef USE_CUDA
TEST(LIBCUDA, poseidon_test2)
{
    // 4 leaves of 7 elements each -> Merkle tree has 7 nodes
    u64 test_leaves[28] = {
        12382199520291307008, 18193113598248284716, 17339479877015319223, 10837159358996869336, 9988531527727040483, 5682487500867411209, 13124187887292514366,
        8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027,
        10465118329878758468, 5866464242232862106, 15506463679657361352, 18404485636523119190, 15311871720566825080, 5967980567132965479, 14180845406393061616,
        15480539652174185186, 5454640537573844893, 3664852224809466446, 5547792914986991141, 5885254103823722535, 6014567676786509263, 11767239063322171808};

    // CPU
    u64 tree1[28] = {0ul};

    for (u32 i = 0; i < 4; i++)
    {
        PoseidonHasher::cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    PoseidonHasher::cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    PoseidonHasher::cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    PoseidonHasher::cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

    // GPU
    u64 tree2[28] = {0ul};

    u64 *gpu_leaf;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_leaf, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_leaf, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon_hash_step1<<<1, 4>>>(gpu_leaf, gpu_hash, 7, 4);
    poseidon_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    poseidon_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, monolith_test3)
{
    u64 inp[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    u64 hash[HASH_SIZE_U64] = {0};
    u64 expected[HASH_SIZE_U64] = {0xCB4EF9B3FE5BCA9E, 0xE03C9506D19C8216, 0x2F05CFB355E880C, 0xF614E84BF4DF8342};

    MonolithHasher::cpu_hash_one(inp, 12, hash);
    for (int i = 0; i < 4; i++)
    {
        assert(hash[i] == expected[i]);
    }

#ifdef USE_CUDA
    u64 *gpu_inp;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_inp, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_inp, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    monolith_hash<<<1, 1>>>(gpu_inp, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(hash, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++)
    {
        assert(hash[i] == expected[i]);
    }
#endif
}

TEST(LIBCUDA, poseidon2_test1)
{
    // similar to the test in goldilocks repo
    // test 1 - Fibonacci
    u64 inp[12];
    inp[0] = 0;
    inp[1] = 1;
    for (int i = 2; i < 12; i++)
    {
        inp[i] = inp[i - 2] + inp[i - 1];
    }

    u64 h1[HASH_SIZE_U64] = {0u};

    Poseidon2Hasher hasher;
    hasher.cpu_hash_one(inp, 12, h1);
#ifdef DEBUG
    printhash(h1);
#endif

    assert(h1[0] == 0x133a03eca11d93fb);
    assert(h1[1] == 0x5365414fb618f58d);
    assert(h1[2] == 0xfa49f50f3a2ba2e5);
    assert(h1[3] == 0xd16e53672c9832a4);

#ifdef USE_CUDA
    u64 h2[HASH_SIZE_U64] = {0u};
    u64 *gpu_inp;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_inp, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_inp, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_hash<<<1, 1>>>(gpu_inp, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(h2, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef DEBUG
    printhash(h2);
#endif

    assert(h2[0] == 0x133a03eca11d93fb);
    assert(h2[1] == 0x5365414fb618f58d);
    assert(h2[2] == 0xfa49f50f3a2ba2e5);
    assert(h2[3] == 0xd16e53672c9832a4);
#endif
}

#ifdef USE_CUDA
TEST(LIBCUDA, poseidon2_test2)
{
    // 4 leaves of 7 elements each -> Merkle tree has 7 nodes
    u64 test_leaves[28] = {
        12382199520291307008, 18193113598248284716, 17339479877015319223, 10837159358996869336, 9988531527727040483, 5682487500867411209, 13124187887292514366,
        8395359103262935841, 1377884553022145855, 2370707998790318766, 3651132590097252162, 1141848076261006345, 12736915248278257710, 9898074228282442027,
        10465118329878758468, 5866464242232862106, 15506463679657361352, 18404485636523119190, 15311871720566825080, 5967980567132965479, 14180845406393061616,
        15480539652174185186, 5454640537573844893, 3664852224809466446, 5547792914986991141, 5885254103823722535, 6014567676786509263, 11767239063322171808};

    // CPU
    u64 tree1[28] = {0ul};

    Poseidon2Hasher hasher;
    for (u32 i = 0; i < 4; i++)
    {
        hasher.cpu_hash_one(test_leaves + 7 * i, 7, tree1 + 4 * i);
    }
    hasher.cpu_hash_two(tree1, tree1 + 4, tree1 + 16);
    hasher.cpu_hash_two(tree1 + 8, tree1 + 12, tree1 + 20);
    hasher.cpu_hash_two(tree1 + 16, tree1 + 20, tree1 + 24);

    // GPU
    u64 tree2[28] = {0ul};

    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 28 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, test_leaves, 28 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidon2_hash_step1<<<1, 4>>>(gpu_data, gpu_hash, 7, 4);
    poseidon2_hash_step2<<<1, 2>>>(gpu_hash, gpu_hash + 16, 2);
    poseidon2_hash_step2<<<1, 2>>>(gpu_hash + 16, gpu_hash + 24, 1);
    CHECKCUDAERR(cudaMemcpy(tree2, gpu_hash, 28 * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < 28; i++)
    {
        assert(tree1[i] == tree2[i]);
    }
}
#endif // USE_CUDA

TEST(LIBCUDA, poseidonbn128_test1)
{
    u64 inp[12] = {8917524657281059100ull,
                   13029010200779371910ull,
                   16138660518493481604ull,
                   17277322750214136960ull,
                   1441151880423231822ull,
                   0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};

    u64 expected[HASH_SIZE_U64] = {2163910501769503938ull, 9976732063159483418ull, 662985512748194034ull, 3626198389901409849ull};

    u64 cpu_out[HASH_SIZE_U64];

    PoseidonBN128Hasher hasher;
    hasher.cpu_hash_one(inp, 12, cpu_out);

#ifdef DEBUG
    printhash(cpu_out);
#endif

    assert(cpu_out[0] == expected[0]);
    assert(cpu_out[1] == expected[1]);
    assert(cpu_out[2] == expected[2]);
    assert(cpu_out[3] == expected[3]);

#ifdef USE_CUDA
    u64 gpu_out[HASH_SIZE_U64];
    u64 *gpu_data;
    u64 *gpu_hash;
    CHECKCUDAERR(cudaMalloc(&gpu_data, 12 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_hash, 4 * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_data, inp, 12 * sizeof(u64), cudaMemcpyHostToDevice));
    poseidonbn128_hash<<<1, 1>>>(gpu_data, gpu_hash, 12);
    CHECKCUDAERR(cudaMemcpy(gpu_out, gpu_hash, 4 * sizeof(u64), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    printhash(gpu_out);
#endif

    assert(gpu_out[0] == expected[0]);
    assert(gpu_out[1] == expected[1]);
    assert(gpu_out[2] == expected[2]);
    assert(gpu_out[3] == expected[3]);
#endif // USE_CUDA
}

#ifdef USE_CUDA
void compare_results(u64 *digests_buf1, u64 *digests_buf2, u32 n_digests, u64 *cap_buf1, u64 *cap_buf2, u32 n_caps)
{
    u64 *ptr1 = digests_buf1;
    u64 *ptr2 = digests_buf2;
#ifdef DEBUG
    for (int i = 0; i < n_digests; i++, ptr1 += HASH_SIZE_U64, ptr2 += HASH_SIZE_U64)
    {
        printf("Hashes digests\n");
        printhash(ptr1);
        printhash(ptr2);
    }
    ptr1 = digests_buf1;
    ptr2 = digests_buf2;
#endif
    for (int i = 0; i < n_digests * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
#ifdef DEBUG
    for (int i = 0; i < n_caps; i++, ptr1 += HASH_SIZE_U64, ptr2 += HASH_SIZE_U64)
    {
        printf("Hashes digests\n");
        printhash(ptr1);
        printhash(ptr2);
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
#endif
    for (int i = 0; i < n_caps * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        assert(*ptr1 == *ptr2);
    }
}
/*
 * Run on GPU and CPU and compare the results. They have to be the same.
 */
#define LOG_SIZE 2
#define LEAF_SIZE_U64 6

TEST(LIBCUDA, merkle_test2)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 n_leaves = (1 << LOG_SIZE);
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));

    // Generate random leaves
    srand(time(NULL));
    for (int i = 0; i < n_leaves; i++)
    {
        for (int j = 0; j < LEAF_SIZE_U64; j++)
        {
            u32 r = rand();
            leaves_buf[i * LEAF_SIZE_U64 + j] = (u64)r << 32 + r * 88958514;
        }
    }
#ifdef DEBUG
    printf("Leaves count: %ld\n", n_leaves);
    printf("Leaf size: %d\n", LEAF_SIZE_U64);
    printf("Digests count: %ld\n", n_digests);
    printf("Caps count: %ld\n", n_caps);
    printf("Caps height: %ld\n", cap_h);
#endif // DEBUG

    // Compute on GPU
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u32 *gpu_caps;

    u64 *digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));

    CHECKCUDAERR(cudaMalloc(&gpu_leaves, n_leaves * LEAF_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64)));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    CHECKCUDAERR(cudaMemcpy(gpu_leaves, leaves_buf, n_leaves * LEAF_SIZE_U64 * sizeof(u64), cudaMemcpyHostToDevice));
    fill_digests_buf_linear_gpu_with_gpu_ptr(
        gpu_digests,
        gpu_caps,
        gpu_leaves,
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashType::HashPoseidon,
        0);
    CHECKCUDAERR(cudaMemcpy(digests_buf2, gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(cap_buf2, gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
#ifdef TIMING
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);
#endif
    CHECKCUDAERR(cudaFree(gpu_leaves));
    CHECKCUDAERR(cudaFree(gpu_digests));
    CHECKCUDAERR(cudaFree(gpu_caps));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, LEAF_SIZE_U64, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

    compare_results(digests_buf1, digests_buf2, n_digests, cap_buf1, cap_buf2, n_caps);

    free(digests_buf1);
    free(digests_buf2);
    free(cap_buf1);
    free(cap_buf2);
}
#endif // USE_CUDA

TEST(LIBCUDA, merkle_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 3651132590097252162ull, 1141848076261006345ull, 12736915248278257710ull, 9898074228282442027ull};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}

#ifdef __USE_AVX__
TEST(LIBCUDA, merkle_avx_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 3651132590097252162ull, 1141848076261006345ull, 12736915248278257710ull, 9898074228282442027ull};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu_avx(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}
#endif // __USE_AVX__

#ifdef __AVX512__
TEST(LIBCUDA, merkle_avx512_test3)
{
#ifdef TIMING
    struct timeval t0, t1;
#endif

    u64 leaf_size = 7;
    u64 n_leaves = 4;
    u64 n_caps = n_leaves;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 cap_h = log2(n_caps);

    // u64 *digests_buf1 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf1 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    u64 *digests_buf1 = cap_buf1;
    u64 *leaves_buf = (u64 *)malloc(n_leaves * leaf_size * sizeof(u64));

    u64 leaf[7] = {8395359103262935841ull, 1377884553022145855ull, 2370707998790318766ull, 3651132590097252162ull, 1141848076261006345ull, 12736915248278257710ull, 9898074228282442027ull};
    memcpy(leaves_buf, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 7, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 14, leaf, 7 * sizeof(u64));
    memcpy(leaves_buf + 21, leaf, 7 * sizeof(u64));

#ifdef TIMING
    gettimeofday(&t0, 0);
#endif
    fill_digests_buf_linear_cpu_avx512(digests_buf1, cap_buf1, leaves_buf, n_digests, n_caps, n_leaves, leaf_size, cap_h, HashType::HashPoseidon);
#ifdef TIMING
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);
#endif

#ifdef DEBUG
    printf("Digests:\n");
    for (int i = 0; i < n_digests; i++)
    {
        printhash(digests_buf1 + i * HASH_SIZE_U64);
    }
    printf("Caps:\n");
    for (int i = 0; i < n_caps; i++)
    {
        printhash(cap_buf1 + i * HASH_SIZE_U64);
    }
#endif
    free(cap_buf1);
    free(leaves_buf);
}
#endif // __AVX512__

#ifdef USE_CUDA
// ---------- PolynomialBatchGPU tests ----------

#include <prover/polynomial_batch.cuh>
#include <ntt/ntt.cuh>
#include <ff/gl64_params.hpp>
#include <utils/all_gpus.hpp>

static void init_gpu_for_poly_batch(u32 max_log_degree)
{
    size_t n_gpus = ngpus();
    for (size_t d = 0; d < n_gpus; d++) {
        auto &gpu = select_gpu(d);
        ntt::init_coset(gpu, max_log_degree, fr_t(GROUP_GENERATOR));
        for (size_t k = 2; k <= max_log_degree; k++) {
            ntt::init_twiddle_factors(gpu, k);
        }
    }
}

TEST(PolynomialBatch, from_coeffs_basic_sizes)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 3;
    const bool BLINDING = false;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 p = 0; p < NUM_POLYS; p++) {
        for (u32 i = 0; i < DEGREE; i++) {
            host_coeffs[p * DEGREE + i] = (u64)(p + 1) * (i + 1) % cpp_gl64_t::MOD;
        }
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    auto batch = PolynomialBatchGPU::from_coeffs(
        gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        BLINDING, CAP_HEIGHT, 0);

    size_t expected_domain = (size_t)1 << (LOG_DEGREE + RATE_BITS);
    ASSERT_EQ(batch.num_leaves, expected_domain);
    ASSERT_EQ(batch.leaf_size, (size_t)NUM_POLYS);
    ASSERT_EQ(batch.cap_len, (size_t)(1 << CAP_HEIGHT));
    ASSERT_EQ(batch.num_digests, 2 * (expected_domain - batch.cap_len));
    ASSERT_NE(batch.lde_gpu, nullptr);
    ASSERT_NE(batch.digests_gpu, nullptr);
    ASSERT_NE(batch.cap_gpu, nullptr);

    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, from_coeffs_with_blinding)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 5;
    const bool BLINDING = true;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 p = 0; p < NUM_POLYS; p++) {
        for (u32 i = 0; i < DEGREE; i++) {
            host_coeffs[p * DEGREE + i] = ((u64)(p * 13 + i * 7 + 42)) % cpp_gl64_t::MOD;
        }
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    auto batch = PolynomialBatchGPU::from_coeffs(
        gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        BLINDING, CAP_HEIGHT, 0);

    ASSERT_EQ(batch.leaf_size, (size_t)(NUM_POLYS + SALT_SIZE));
    ASSERT_EQ(batch.blinding, true);

    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, from_coeffs_merkle_cap_nonzero)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 2;
    const u32 NUM_POLYS = 4;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 i = 0; i < NUM_POLYS * DEGREE; i++) {
        host_coeffs[i] = ((u64)i * 12345 + 67890) % cpp_gl64_t::MOD;
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    auto batch = PolynomialBatchGPU::from_coeffs(
        gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        false, CAP_HEIGHT, 0);

    size_t cap_elems = batch.cap_len * NUM_HASH_OUT_ELTS;
    std::vector<u64> host_cap(cap_elems, 0);
    batch.copy_cap_to_host((fr_t *)host_cap.data(), cap_elems);

    bool all_zero = true;
    for (size_t i = 0; i < cap_elems; i++) {
        if (host_cap[i] != 0) { all_zero = false; break; }
    }
    ASSERT_FALSE(all_zero) << "Merkle cap should contain non-zero hash values";

    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, from_coeffs_merkle_cpu_gpu_match)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 2;
    const size_t DOMAIN_SIZE = (size_t)1 << (LOG_DEGREE + RATE_BITS);

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 p = 0; p < NUM_POLYS; p++) {
        for (u32 i = 0; i < DEGREE; i++) {
            host_coeffs[p * DEGREE + i] = ((u64)(p * 100 + i)) % cpp_gl64_t::MOD;
        }
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    // Build on GPU via PolynomialBatchGPU
    auto batch = PolynomialBatchGPU::from_coeffs(
        gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        false, CAP_HEIGHT, 0);

    // Copy LDE leaves back from GPU
    size_t total_leaf_elems = DOMAIN_SIZE * NUM_POLYS;
    std::vector<u64> gpu_leaves(total_leaf_elems);
    CHECKCUDAERR(cudaMemcpy(gpu_leaves.data(), batch.lde_gpu,
                            total_leaf_elems * sizeof(u64),
                            cudaMemcpyDeviceToHost));

    // Build Merkle tree on CPU from the same leaves for comparison
    u64 n_caps_cpu = (u64)1 << CAP_HEIGHT;
    u64 n_digests_cpu = 2 * (DOMAIN_SIZE - n_caps_cpu);

    std::vector<u64> cpu_digests(n_digests_cpu * HASH_SIZE_U64, 0);
    std::vector<u64> cpu_cap(n_caps_cpu * HASH_SIZE_U64, 0);

    fill_digests_buf_linear_cpu(
        cpu_digests.data(), cpu_cap.data(), gpu_leaves.data(),
        n_digests_cpu, n_caps_cpu, DOMAIN_SIZE,
        NUM_POLYS, CAP_HEIGHT, HashType::HashPoseidon);

    // Copy GPU Merkle results to host
    std::vector<u64> gpu_digests(n_digests_cpu * HASH_SIZE_U64, 0);
    std::vector<u64> gpu_cap(n_caps_cpu * HASH_SIZE_U64, 0);
    batch.copy_digests_to_host((fr_t *)gpu_digests.data(), n_digests_cpu * HASH_SIZE_U64);
    batch.copy_cap_to_host((fr_t *)gpu_cap.data(), n_caps_cpu * HASH_SIZE_U64);

    // Compare caps
    for (size_t i = 0; i < n_caps_cpu * HASH_SIZE_U64; i++) {
        ASSERT_EQ(cpu_cap[i], gpu_cap[i])
            << "Cap mismatch at element " << i;
    }

    // Compare digests
    for (size_t i = 0; i < n_digests_cpu * HASH_SIZE_U64; i++) {
        ASSERT_EQ(cpu_digests[i], gpu_digests[i])
            << "Digest mismatch at element " << i;
    }

    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, from_values_basic)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 2;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_values[NUM_POLYS * DEGREE];
    for (u32 p = 0; p < NUM_POLYS; p++) {
        for (u32 i = 0; i < DEGREE; i++) {
            host_values[p * DEGREE + i] = ((u64)(p + 1) * (i + 1) * 31) % cpp_gl64_t::MOD;
        }
    }

    fr_t *gpu_values;
    CHECKCUDAERR(cudaMalloc(&gpu_values, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_values, host_values,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    auto batch = PolynomialBatchGPU::from_values(
        gpu_values, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        false, CAP_HEIGHT, 0);

    size_t expected_domain = (size_t)1 << (LOG_DEGREE + RATE_BITS);
    ASSERT_EQ(batch.num_leaves, expected_domain);
    ASSERT_EQ(batch.leaf_size, (size_t)NUM_POLYS);
    ASSERT_NE(batch.lde_gpu, nullptr);
    ASSERT_NE(batch.cap_gpu, nullptr);

    // Coefficients pointer should be the same as the input (IFFT was in-place)
    ASSERT_EQ(batch.coeffs_gpu, gpu_values);

    cudaFree(gpu_values);
}

TEST(PolynomialBatch, lde_values_offset_bit_reversal)
{
    PolynomialBatchGPU batch;
    batch.degree_log = 3;
    batch.rate_bits = 2;
    batch.leaf_size = 5;

    size_t step = 1 << (batch.rate_bits);

    // index=0, step=4 -> raw=0, reversed=0 -> offset = 0*5 = 0
    ASSERT_EQ(batch.lde_values_offset(0, step), (size_t)0);

    // index=1, step=4 -> raw=4, bit-reverse 4 (=00100) in 5 bits -> 00100 reversed = 00100 = 4
    // Actually: 4 in binary is 00100, reversed in 5 bits = 00100 = 4
    size_t offset1 = batch.lde_values_offset(1, step);
    ASSERT_EQ(offset1, 4 * 5);

    // index=2, step=4 -> raw=8, bit-reverse 8 (=01000) in 5 bits = 00010 = 2
    size_t offset2 = batch.lde_values_offset(2, step);
    ASSERT_EQ(offset2, 2 * 5);
}

TEST(PolynomialBatch, move_semantics)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 2;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 i = 0; i < NUM_POLYS * DEGREE; i++) {
        host_coeffs[i] = (u64)(i + 1) % cpp_gl64_t::MOD;
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    auto batch1 = PolynomialBatchGPU::from_coeffs(
        gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS,
        false, CAP_HEIGHT, 0);

    fr_t *original_lde = batch1.lde_gpu;
    fr_t *original_cap = batch1.cap_gpu;

    // Move construct
    PolynomialBatchGPU batch2(std::move(batch1));

    ASSERT_EQ(batch1.lde_gpu, nullptr);
    ASSERT_EQ(batch1.cap_gpu, nullptr);
    ASSERT_EQ(batch2.lde_gpu, original_lde);
    ASSERT_EQ(batch2.cap_gpu, original_cap);
    ASSERT_EQ(batch2.num_polynomials, (size_t)NUM_POLYS);

    // Move assign
    PolynomialBatchGPU batch3;
    batch3 = std::move(batch2);

    ASSERT_EQ(batch2.lde_gpu, nullptr);
    ASSERT_EQ(batch3.lde_gpu, original_lde);
    ASSERT_EQ(batch3.num_leaves, (size_t)(1 << (LOG_DEGREE + RATE_BITS)));

    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, c_api_from_coeffs)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 3;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_coeffs[NUM_POLYS * DEGREE];
    for (u32 i = 0; i < NUM_POLYS * DEGREE; i++) {
        host_coeffs[i] = ((u64)i * 997 + 1) % cpp_gl64_t::MOD;
    }

    fr_t *gpu_coeffs;
    CHECKCUDAERR(cudaMalloc(&gpu_coeffs, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_coeffs, host_coeffs,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    void *out_lde = nullptr, *out_digests = nullptr, *out_cap = nullptr;
    uint64_t out_num_leaves = 0, out_leaf_size = 0, out_num_digests = 0, out_cap_len = 0;

    RustError err = polynomial_batch_from_coeffs(
        0, gpu_coeffs, NUM_POLYS, LOG_DEGREE, RATE_BITS, 0, CAP_HEIGHT,
        &out_lde, &out_digests, &out_cap,
        &out_num_leaves, &out_leaf_size, &out_num_digests, &out_cap_len);

    ASSERT_EQ(err.code, 0);
    ASSERT_NE(out_lde, nullptr);
    ASSERT_NE(out_digests, nullptr);
    ASSERT_NE(out_cap, nullptr);
    ASSERT_EQ(out_num_leaves, (uint64_t)(1 << (LOG_DEGREE + RATE_BITS)));
    ASSERT_EQ(out_leaf_size, (uint64_t)NUM_POLYS);
    ASSERT_EQ(out_cap_len, (uint64_t)(1 << CAP_HEIGHT));

    cudaFree(out_lde);
    cudaFree(out_digests);
    cudaFree(out_cap);
    cudaFree(gpu_coeffs);
}

TEST(PolynomialBatch, c_api_from_values)
{
    const u32 LOG_DEGREE = 4;
    const u32 DEGREE = 1 << LOG_DEGREE;
    const u32 RATE_BITS = 3;
    const u32 CAP_HEIGHT = 1;
    const u32 NUM_POLYS = 2;

    init_gpu_for_poly_batch(LOG_DEGREE + RATE_BITS);

    u64 host_values[NUM_POLYS * DEGREE];
    for (u32 i = 0; i < NUM_POLYS * DEGREE; i++) {
        host_values[i] = ((u64)i * 53 + 7) % cpp_gl64_t::MOD;
    }

    fr_t *gpu_values;
    CHECKCUDAERR(cudaMalloc(&gpu_values, NUM_POLYS * DEGREE * sizeof(fr_t)));
    CHECKCUDAERR(cudaMemcpy(gpu_values, host_values,
                            NUM_POLYS * DEGREE * sizeof(fr_t),
                            cudaMemcpyHostToDevice));

    void *out_lde = nullptr, *out_digests = nullptr, *out_cap = nullptr, *out_coeffs = nullptr;
    uint64_t out_num_leaves = 0, out_leaf_size = 0, out_num_digests = 0, out_cap_len = 0;

    RustError err = polynomial_batch_from_values(
        0, gpu_values, NUM_POLYS, LOG_DEGREE, RATE_BITS, 0, CAP_HEIGHT,
        &out_lde, &out_digests, &out_cap, &out_coeffs,
        &out_num_leaves, &out_leaf_size, &out_num_digests, &out_cap_len);

    ASSERT_EQ(err.code, 0);
    ASSERT_NE(out_lde, nullptr);
    ASSERT_NE(out_cap, nullptr);
    ASSERT_EQ(out_num_leaves, (uint64_t)(1 << (LOG_DEGREE + RATE_BITS)));
    ASSERT_EQ(out_leaf_size, (uint64_t)NUM_POLYS);
    // coeffs pointer should be the same as input (IFFT in-place)
    ASSERT_EQ(out_coeffs, (void *)gpu_values);

    cudaFree(out_lde);
    cudaFree(out_digests);
    cudaFree(out_cap);
    cudaFree(gpu_values);
}

#endif // USE_CUDA

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}