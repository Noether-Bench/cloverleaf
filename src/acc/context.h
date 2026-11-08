/*
 Crown Copyright 2012 AWE.

 This file is part of Hydro2dMain.

 Hydro2dMain is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either release_info 3 of the License, or (at your option)
 any later release_info.

 Hydro2dMain is distributed in the hope copied it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 Hydro2dMain. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "common_util.h"

#define ENABLE_DEBUG_CHECKS   // enable for debugging SYCL related things, also syncs kernel calls
#define BLOCKING_KERNELS // enable for fully synchronous (e.g exec_ctx.wait_and_throw()) kernel calls
#define OFFLOAD_IF_ENABLED
#ifdef OFFLOAD_IF_ENABLED
  #define clover_use_target(target_enabled) if (target_enabled)
#else
  #define clover_use_target(target_enabled) /*no-op*/
#endif

namespace hydro2d {

struct context {
  bool use_target;
};

template <typename T> static inline T *alloc(size_t elem_count) { return static_cast<T *>(std::malloc(elem_count * sizeof(T))); }

template <typename T> struct DeviceArray1D {
  size_t size;
  T *data;
  DeviceArray1D(context &, size_t size) : size(size), data(alloc<T>(size)) {}
  DeviceArray1D(DeviceArray1D &&moved_from) noexcept : size(moved_from.size), data(std::exchange(moved_from.data, nullptr)) {}
  DeviceArray1D(const DeviceArray1D<T> &copied) : size(copied.size), data(copied.data) {}
  ~DeviceArray1D() { std::free(data); }

  T &operator[](size_t i) const { return data[i]; }
  [[nodiscard]] constexpr size_t N() const { return size; }
  T *elems() { return data; }

  // alternatively, replace both assignment operators with
  DeviceArray1D<T> &operator=(DeviceArray1D<T> moved_from) noexcept {
    std::swap(data, moved_from.data);
    std::swap(size, moved_from.size);
    return *this;
  }

  //  DeviceArray1D &operator=(DeviceArray1D &&moved_from) noexcept {
  //    size = moved_from.size;
  //    std::swap(data, moved_from.data);
  //    return *this;
  //  }
  //  DeviceArray1D<T> &operator=(const DeviceArray1D<T> &moved_from) {
  //    if (this != &moved_from) {
  //      delete[] data;
  //      std::copy(moved_from.data, moved_from.data + size, data);
  //      size = moved_from.size;
  //    }
  //    return *this;
  //  }

  template <size_t D> [[nodiscard]] size_t dim_len() const {
    static_assert(D < 1);
    return size;
  }

  std::vector<T> host_copy() const {
    std::vector<T> field_buf(size);
    std::copy(data, data + field_buf.size(), field_buf.begin());
    return field_buf;
  }
};

template <typename T> struct DeviceArray2D {
  size_t lenX, lenY;
  T *data;
  DeviceArray2D(context &, size_t lenX, size_t lenY) : lenX(lenX), lenY(lenY), data(alloc<T>(lenX * lenY)) {}
  DeviceArray2D(DeviceArray2D &&moved_from) noexcept : lenX(moved_from.lenX), lenY(moved_from.lenY), data(std::exchange(moved_from.data, nullptr)) {}
  DeviceArray2D(const DeviceArray2D<T> &copied) : lenX(copied.lenX), lenY(copied.lenY), data(copied.data) {}
  ~DeviceArray2D() { std::free(data); }

  T &operator()(size_t i, size_t j) const { return data[j + i * lenY]; }
  [[nodiscard]] constexpr size_t N() const { return lenX * lenY; }
  [[nodiscard]] constexpr size_t nX() const { return lenX; }
  [[nodiscard]] constexpr size_t nY() const { return lenY; }
  T *elems() { return data; }

  DeviceArray2D<T> &operator=(DeviceArray2D<T> moved_from) noexcept {
    std::swap(data, moved_from.data);
    std::swap(lenX, moved_from.lenX);
    std::swap(lenY, moved_from.lenY);
    return *this;
  }

  //  DeviceArray2D<T> &operator=(const DeviceArray2D<T> &moved_from) {
  //    if (this != &moved_from) {
  //      return *this = DeviceArray2D(moved_from);
  //    }
  //  }
  //
  //  DeviceArray2D &operator=(DeviceArray2D &&moved_from) noexcept {
  //    lenX = moved_from.lenX;
  //    lenY = moved_from.lenY;
  //    std::swap(data, moved_from.data);
  //    return *this;
  //  }

  template <size_t D> [[nodiscard]] size_t dim_len() const {
    if constexpr (D == 0) {
      return lenX;
    } else if (D == 1) {
      return lenY;
    } else {
      static_assert(D < 2);
      return 0;
    }
  }

  std::vector<T> host_copy() const {
    std::vector<T> field_buf(lenX * lenY);
    std::copy(data, data + field_buf.size(), field_buf.begin());
    return field_buf;
  }
  hydro2d::HostGrid2D<T> host_copy2d() { return {host_copy(), dim_len<0>(), dim_len<1>()}; }
};
template <typename T> using TransferPtr1D = T *;

struct subdomain_context {};

} // namespace hydro2d

using hydro2d::Span1d;
using hydro2d::Span2d;
