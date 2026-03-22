// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_UTILS_HOST_BIT_HPP
#define ZEKNOX_UTILS_HOST_BIT_HPP

#include <climits>
#include <cstddef>

#if defined(__GNUC__) || defined(__clang__)
inline std::size_t host_lg2(std::size_t n)
{
    if (n <= 1) {
        return 0;
    }
    const int w = sizeof(unsigned long long) * CHAR_BIT;
    const unsigned long long x = static_cast<unsigned long long>(n);
    return static_cast<std::size_t>(w - __builtin_clzll(x - 1ULL));
}
#else
inline std::size_t host_lg2(std::size_t n)
{
    std::size_t l = 0;
    while (((std::size_t)1 << l) < n) {
        ++l;
    }
    return l;
}
#endif

#endif
