

#include "halo_exchange.h"
#include "device_context.h"

using edge_buffer = hydro2d::DeviceArray1D<double>;
using cell_array = hydro2d::DeviceArray2D<double>;

static void stagger_offsets(int field_set, int cell_centred, int node_centred, int xface_centred, int y_face_centred, int &x_extra,
                            int &y_extra) {
  x_extra = 0;
  y_extra = 0;
  if (field_set == cell_centred) {
    x_extra = 0;
    y_extra = 0;
  }
  if (field_set == node_centred) {
    x_extra = 1;
    y_extra = 1;
  }
  if (field_set == xface_centred) {
    x_extra = 1;
    y_extra = 0;
  }
  if (field_set == y_face_centred) {
    x_extra = 0;
    y_extra = 1;
  }
}

void hydro2d_pack_edge_left(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, cell_array &src_grid,
                              edge_buffer &left_snd_buffer, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                              int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);


  double *left_snd = left_snd_buffer.data;
  double *field = src_grid.data;
  const size_t field_nx = src_grid.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = (y_lo - halo_width + 1); k < (y_hi + y_extra + halo_width + 2); k++) {
    for (int j = 0; j < halo_width; ++j) {
      int buf_pos = slot_start + j + k * halo_width;
      left_snd[buf_pos] = field[(x_lo + x_extra - 1 + j + 2) + (k)*field_nx];
    }
  }
}

void hydro2d_pack_edge_right(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               cell_array &src_grid, edge_buffer &right_snd_buffer, int cell_centred,
                               int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  double *right_snd = right_snd_buffer.data;
  double *field = src_grid.data;
  const size_t field_nx = src_grid.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = (y_lo - halo_width + 1); k < (y_hi + y_extra + halo_width + 2); k++) {
    for (int j = 0; j < halo_width; ++j) {
      int buf_pos = slot_start + j + k * halo_width;
      right_snd[buf_pos] = field[(x_hi + 1 - j) + (k)*field_nx];
    }
  }
}

void hydro2d_unpack_edge_left(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                                cell_array &src_grid, edge_buffer &left_rcv_buffer, int cell_centred,
                                int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);


  double *field = src_grid.data;
  const size_t field_nx = src_grid.nX();
  double *left_rcv = left_rcv_buffer.data;
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = (y_lo - halo_width + 1); k < (y_hi + y_extra + halo_width + 2); k++) {
    for (int j = 0; j < halo_width; ++j) {
      int buf_pos = slot_start + j + k * halo_width;
      field[(x_lo - j) + (k)*field_nx] = left_rcv[buf_pos];
    }
  }
}

void hydro2d_unpack_edge_right(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                                 cell_array &src_grid, edge_buffer &right_rcv_buffer, int cell_centred,
                                 int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  double *right_rcv = right_rcv_buffer.data;
  double *field = src_grid.data;
  const size_t field_nx = src_grid.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = (y_lo - halo_width + 1); k < (y_hi + y_extra + halo_width + 2); k++) {
    for (int j = 0; j < halo_width; ++j) {
      int buf_pos = slot_start + j + k * halo_width;
      field[(x_hi + x_extra + j + 2) + (k)*field_nx] = right_rcv[buf_pos];
    }
  }
}

void hydro2d_pack_edge_bottom(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                                cell_array &src_grid, edge_buffer &bottom_snd_buffer, int cell_centred,
                                int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  for (int k = 0; k < halo_width; ++k) {

    double *bottom_snd = bottom_snd_buffer.data;
    double *field = src_grid.data;
    const size_t field_nx = src_grid.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + x_extra + halo_width + 2); j++) {
      int buf_pos = slot_start + k + j * halo_width;
      bottom_snd[buf_pos] = field[j + (y_lo + y_extra - 1 + k + 2) * field_nx];
    }
  }
}

void hydro2d_pack_edge_top(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, cell_array &src_grid,
                             edge_buffer &top_snd_buffer, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                             int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  for (int k = 0; k < halo_width; ++k) {

    double *top_snd = top_snd_buffer.data;
    double *field = src_grid.data;
    const size_t field_nx = src_grid.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + x_extra + halo_width + 2); j++) {
      int buf_pos = slot_start + k + j * halo_width;
      top_snd[buf_pos] = field[j + (y_hi + 1 - k) * field_nx];
    }
  }
}

void hydro2d_unpack_edge_bottom(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                                  cell_array &src_grid, edge_buffer &bottom_rcv_buffer, int cell_centred,
                                  int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  for (int k = 0; k < halo_width; ++k) {

    double *field = src_grid.data;
    const size_t field_nx = src_grid.nX();
    double *bottom_rcv = bottom_rcv_buffer.data;
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + x_extra + halo_width + 2); j++) {
      int buf_pos = slot_start + k + j * halo_width;
      field[j + (y_lo - k) * field_nx] = bottom_rcv[buf_pos];
    }
  }
}

void hydro2d_unpack_edge_top(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               cell_array &src_grid, edge_buffer &top_rcv_buffer, int cell_centred,
                               int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {


  int x_extra, y_extra;
  stagger_offsets(field_set, cell_centred, node_centred, xface_centred, y_face_centred, x_extra, y_extra);

  for (int k = 0; k < halo_width; ++k) {

    double *field = src_grid.data;
    const size_t field_nx = src_grid.nX();
    double *top_rcv = top_rcv_buffer.data;
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + x_extra + halo_width + 2); j++) {
      int buf_pos = slot_start + k + j * halo_width;
      field[j + (y_hi + y_extra + k + 2) * field_nx] = top_rcv[buf_pos];
    }
  }
}
