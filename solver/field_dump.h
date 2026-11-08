
#pragma once

#include <cassert>
#include <cstddef>
#include <ostream>

struct sim_state;

namespace hydro2d {

template <typename T> struct HostGrid2D {
  size_t lenX, lenY;
  std::vector<T> elems;
  HostGrid2D(std::vector<T> elems, size_t lenX, size_t lenY);
  T &operator()(size_t i, size_t j)   { return elems[j + i * lenY]; }
};

template <typename T>
HostGrid2D<T>::HostGrid2D(std::vector<T> elems, size_t lenX, size_t lenY) : lenX(lenX), lenY(lenY), elems(elems) {
  if (lenX * lenY != elems.size()) throw std::logic_error("Bad mirror size");
}


struct Span1d {
  const size_t from, to;
  const size_t size;
  template <typename A, typename B> Span1d(A from, B to) : from(from), to(to), size(to - from) {
    assert(from < to);
    assert(size != 0);
  }
  friend std::ostream &operator<<(std::ostream &os, const Span1d &d);
};

struct Span2d {
  const size_t startX, toX;
  const size_t startY, toY;
  const size_t lenX, lenY;
  template <typename A, typename B, typename C, typename D> Span2d(A startX, B startY, C toX, D toY);
  friend std::ostream &operator<<(std::ostream &os, const Span2d &d);
};

template <typename A, typename B, typename C, typename D>
Span2d::Span2d(A startX, B startY, C toX, D toY)
    : startX(startX), toX(toX), startY(startY), toY(toY), lenX(toX - startX), lenY(toY - startY) {
  assert(startX < toX);
  assert(startY < toY);
  assert(lenX != 0);
  assert(lenY != 0);
}

void write_snapshot(sim_state &g, const std::string &out_path);

}
