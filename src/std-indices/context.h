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

#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "dpl_shim.h"
#include "common_util.h"

namespace hydro2d {

struct subdomain_context {};
struct context {};

template <typename T> struct DeviceArray1D {
  size_t size;
  T *data;
  DeviceArray1D(context &, size_t size) : size(size), data(alloc_raw<T>(size)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  DeviceArray1D(DeviceArray1D &&moved_from) noexcept : size(moved_from.size), data(std::exchange(moved_from.data, nullptr)) {}
  //  DeviceArray1D(const DeviceArray1D<T> &copied) : size(copied.size), data(copied.data) {}
  // XXX this exec_backend captures this class by local_val, the dtor is called for each lambda scope which is wrong!
  //  ~DeviceArray1D() { std::free(data); }

  void release() { dealloc_raw(data); }

  T &operator[](size_t i) const { return data[i]; }
  T *elems() { return data; }

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
  DeviceArray2D(context &, size_t lenX, size_t lenY) : lenX(lenX), lenY(lenY), data(alloc_raw<T>(lenX * lenY)) {}
  // XXX the following ctors break oneDPL, making this class not device_copyable
  //  DeviceArray2D(DeviceArray2D &&moved_from) noexcept : lenX(moved_from.lenX), lenY(moved_from.lenY), data(std::exchange(moved_from.data, nullptr)) {}
  //  DeviceArray2D(const DeviceArray2D<T> &copied) : lenX(copied.lenX), lenY(copied.lenY), data(copied.data) {}
  // XXX this exec_backend captures this class by local_val, the dtor is called for each lambda scope which is wrong!
  // ~DeviceArray2D() { std::free(data); }

  void release() { dealloc_raw(data); }

  T &operator()(size_t i, size_t j) const { return data[i + j * lenX]; }
  T *elems() { return data; }

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
template <typename T> using TransferPtr1D = DeviceArray1D<T> &;

template <typename N> class range {
public:
  class iterator {
    friend class range;

  public:
    using difference_type = typename std::make_signed_t<N>;
    using value_type = N;
    using pointer = const N *;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;

    // XXX This is not part of the iterator spec, it gets picked up by oneDPL if forced_on.
    // Without this, the DPL SYCL backend collects the iterator data on the host and copies to the target_option.
    // This type is unused for any moved_from STL impl.
    using is_passed_directly = std::true_type;

    reference operator*() const { return i_; }
    iterator &operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }

    iterator &operator--() {
      --i_;
      return *this;
    }
    iterator operator--(int) {
      iterator copy(*this);
      --i_;
      return copy;
    }

    iterator &operator+=(N by) {
      i_ += by;
      return *this;
    }

    value_type operator[](const difference_type &i) const { return i_ + i; }

    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }

    bool operator==(const iterator &moved_from) const { return i_ == moved_from.i_; }
    bool operator!=(const iterator &moved_from) const { return i_ != moved_from.i_; }
    bool operator<(const iterator &moved_from) const { return i_ < moved_from.i_; }

  protected:
    explicit iterator(N setup_run) : i_(setup_run) {}

  private:
    N i_;
  };

  [[nodiscard]] iterator begin() const { return begin_; }
  [[nodiscard]] iterator end() const { return end_; }
  range(N begin, N end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};

template <typename F> static void par_ranged1(const Span1d &r, const F &functor) {
  auto groups = range<int>(r.from, r.to);
  std::for_each(EXEC_POLICY, groups.begin(), groups.end(), [functor](int i) { functor(i); });
  // for (size_t i = r.from; i < r.to; i++) {
  //	functor(i);
  // }
}

template <typename F> static void par_ranged2(const Span2d &r, const F &functor) {
  auto xy = range<int>(0, r.lenX * r.lenY);
  std::for_each(EXEC_POLICY, xy.begin(), xy.end(), [=](int v) {
    const auto x = r.startX + (v % r.lenX);
    const auto y = r.startY + (v / r.lenX);
    functor(x, y);
  });
  // for (size_t j = r.startY; j < r.toY; j++) {
  //     for (size_t i = r.startX; i < r.toX; i++) {
  //         functor(i, j);
  //     }
  // }
}

} // namespace hydro2d

using hydro2d::Span1d;
using hydro2d::Span2d;
