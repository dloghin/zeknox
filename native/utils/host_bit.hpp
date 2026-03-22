// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ZEKNOX_UTILS_HOST_BIT_HPP
#define ZEKNOX_UTILS_HOST_BIT_HPP

#include <cstddef>

inline std::size_t host_lg2(std::size_t n)
{
    std::size_t l = 0;
    while (((std::size_t)1 << l) < n) {
        ++l;
    }
    return l;
}

#endif
