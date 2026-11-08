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

//  @brief Fortran mpi field_buf packing kernel
//  @author Wayne Gaudin
//  @details Packs/unpacks mpi send and receive buffers

#include "halo_pack.h"

void hydro2d_pack_edge_left(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                              hydro2d::DeviceArray1D<double> &left_snd, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                              int halo_width, int field_set, int slot_start) {

  // Pack

  int x_extra = 0, y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
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

  // DO k=y_lo-halo_width,y_hi+y_extra+halo_width

  hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [&](int k) {
    for (int j = 0; j < halo_width; ++j) {
      size_t buf_pos = slot_start + j + k * halo_width;
      left_snd[buf_pos] = field(x_lo + x_extra - 1 + j + 2, k);
    }
  });
}

void hydro2d_unpack_edge_left(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                hydro2d::DeviceArray1D<double> &left_rcv, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                                int halo_width, int field_set, int slot_start) {

  // Upnack

  int y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
  if (field_set == cell_centred) {
    y_extra = 0;
  }
  if (field_set == node_centred) {
    y_extra = 1;
  }
  if (field_set == xface_centred) {
    y_extra = 0;
  }
  if (field_set == y_face_centred) {
    y_extra = 1;
  }

  // DO k=y_lo-halo_width,y_hi+y_extra+halo_width

  hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [&](int k) {
    for (int j = 0; j < halo_width; ++j) {
      size_t buf_pos = slot_start + j + k * halo_width;
      field(x_lo - j, k) = left_rcv[buf_pos];
    }
  });
}

void hydro2d_pack_edge_right(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                               hydro2d::DeviceArray1D<double> &right_snd, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                               int halo_width, int field_set, int slot_start) {

  // Pack

  int y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
  if (field_set == cell_centred) {
    y_extra = 0;
  }
  if (field_set == node_centred) {
    y_extra = 1;
  }
  if (field_set == xface_centred) {
    y_extra = 0;
  }
  if (field_set == y_face_centred) {
    y_extra = 1;
  }

  // DO k=y_lo-halo_width,y_hi+y_extra+halo_width

  hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [&](int k) {
    for (int j = 0; j < halo_width; ++j) {
      size_t buf_pos = slot_start + j + k * halo_width;
      right_snd[buf_pos] = field(x_hi + 1 - j, k);
    }
  });
}

void hydro2d_unpack_edge_right(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                 hydro2d::DeviceArray1D<double> &right_rcv, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                                 int halo_width, int field_set, int slot_start) {

  // Upnack

  int x_extra = 0, y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
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

  // DO k=y_lo-halo_width,y_hi+y_extra+halo_width

  hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [&](int k) {
    for (int j = 0; j < halo_width; ++j) {
      size_t buf_pos = slot_start + j + k * halo_width;
      field(x_hi + x_extra + j + 2, k) = right_rcv[buf_pos];
    }
  });
}

void hydro2d_pack_edge_top(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                             hydro2d::DeviceArray1D<double> &top_snd, int cell_centred, int node_centred, int xface_centred, int y_face_centred, int halo_width,
                             int field_set, int slot_start) {

  // Pack

  int x_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
  if (field_set == cell_centred) {
    x_extra = 0;
  }
  if (field_set == node_centred) {
    x_extra = 1;
  }
  if (field_set == xface_centred) {
    x_extra = 1;
  }
  if (field_set == y_face_centred) {
    x_extra = 0;
  }

  for (int k = 0; k < halo_width; ++k) {
    // DO j=x_lo-halo_width,x_hi+x_extra+halo_width

    hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [&](int j) {
      size_t buf_pos = slot_start + k + j * halo_width;
      top_snd[buf_pos] = field(j, y_hi + 1 - k);
    });
  }
}

void hydro2d_unpack_edge_top(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                               hydro2d::DeviceArray1D<double> &top_rcv, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                               int halo_width, int field_set, int slot_start) {

  // Unpack

  int x_extra = 0, y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
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

  for (int k = 0; k < halo_width; ++k) {
    // DO j=x_lo-halo_width,x_hi+x_extra+halo_width

    hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [&](int j) {
      size_t buf_pos = slot_start + k + j * halo_width;
      field(j, y_hi + y_extra + k + 2) = top_rcv[buf_pos];
    });
  }
}

void hydro2d_pack_edge_bottom(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                hydro2d::DeviceArray1D<double> &bottom_snd, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                                int halo_width, int field_set, int slot_start) {

  // Pack

  int x_extra = 0, y_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
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

  for (int k = 0; k < halo_width; ++k) {
    // DO j=x_lo-halo_width,x_hi+x_extra+halo_width

    hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [&](int j) {
      size_t buf_pos = slot_start + k + j * halo_width;
      bottom_snd[buf_pos] = field(j, y_lo + y_extra - 1 + k + 2);
    });
  }
}

void hydro2d_unpack_edge_bottom(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                  hydro2d::DeviceArray1D<double> &bottom_rcv, int cell_centred, int node_centred, int xface_centred, int y_face_centred,
                                  int halo_width, int field_set, int slot_start) {

  // Unpack

  int x_extra = 0;

  // These array modifications still need to be added on, plus the source_cell data origin_site changes as in refresh_ghost_cells
  if (field_set == cell_centred) {
    x_extra = 0;
  }
  if (field_set == node_centred) {
    x_extra = 1;
  }
  if (field_set == xface_centred) {
    x_extra = 1;
  }
  if (field_set == y_face_centred) {
    x_extra = 0;
  }

  for (int k = 0; k < halo_width; ++k) {
    // DO j=x_lo-halo_width,x_hi+x_extra+halo_width

    hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [&](int j) {
      size_t buf_pos = slot_start + k + j * halo_width;
      field(j, y_lo - k) = bottom_rcv[buf_pos];
    });
  }
}
