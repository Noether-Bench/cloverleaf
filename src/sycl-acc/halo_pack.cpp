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
#include "context.h"

void hydro2d_pack_edge_left(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &src_grid,
                              hydro2d::DeviceArray1D<double> &left_snd_buffer_buffer, int cell_centred, int node_centred, int xface_centred,
                              int y_face_centred, int halo_width, int field_set, int slot_start) {

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

  hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
    auto field = src_grid.access<R>(h);
    auto left_snd_buffer = left_snd_buffer_buffer.access<W>(h);
    hydro2d::par_ranged<class CONCAT_LINE(hydro2d_pack_edge_left)>(h, {y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [=](id<1> k) {
      for (int j = 0; j < halo_width; ++j) {
        size_t buf_pos = slot_start + j + k.get(0) * halo_width;
        left_snd_buffer[buf_pos] = field[x_lo + x_extra - 1 + j + 2][k.get(0)];
      }
    });
  });
}

void hydro2d_unpack_edge_left(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi,
                                hydro2d::DeviceArray2D<double> &src_grid, hydro2d::DeviceArray1D<double> &left_rcv_buffer_buffer, int cell_centred,
                                int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {

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

  hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
    auto field = src_grid.access<W>(h);
    auto left_rcv_buffer = left_rcv_buffer_buffer.access<R>(h);
    hydro2d::par_ranged<class CONCAT_LINE(hydro2d_unpack_edge_left)>(h, {y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [=](id<1> k) {
      for (int j = 0; j < halo_width; ++j) {
        size_t buf_pos = slot_start + j + k.get(0) * halo_width;
        field[x_lo - j][k.get(0)] = left_rcv_buffer[buf_pos];
      }
    });
  });
}

void hydro2d_pack_edge_right(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &src_grid,
                               hydro2d::DeviceArray1D<double> &right_snd_buffer_buffer, int cell_centred, int node_centred, int xface_centred,
                               int y_face_centred, int halo_width, int field_set, int slot_start) {

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
  hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
    auto field = src_grid.access<R>(h);
    auto right_snd_buffer = right_snd_buffer_buffer.access<W>(h);
    hydro2d::par_ranged<class CONCAT_LINE(hydro2d_pack_edge_right)>(h, {y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [=](id<1> k) {
      for (int j = 0; j < halo_width; ++j) {
        size_t buf_pos = slot_start + j + k.get(0) * halo_width;
        right_snd_buffer[buf_pos] = field[x_hi + 1 - j][k.get(0)];
      }
    });
  });
}

void hydro2d_unpack_edge_right(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi,
                                 hydro2d::DeviceArray2D<double> &src_grid, hydro2d::DeviceArray1D<double> &right_rcv_buffer_buffer, int cell_centred,
                                 int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {

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

  hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
    auto field = src_grid.access<W>(h);
    auto right_rcv_buffer = right_rcv_buffer_buffer.access<R>(h);
    hydro2d::par_ranged<class CONCAT_LINE(hydro2d_unpack_edge_right)>(h, {y_lo - halo_width + 1, y_hi + y_extra + halo_width + 2}, [=](id<1> k) {
      for (int j = 0; j < halo_width; ++j) {
        size_t buf_pos = slot_start + j + k.get(0) * halo_width;
        field[x_hi + x_extra + j + 2][k.get(0)] = right_rcv_buffer[buf_pos];
      }
    });
  });
}

void hydro2d_pack_edge_top(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &src_grid,
                             hydro2d::DeviceArray1D<double> &top_snd_buffer_buffer, int cell_centred, int node_centred, int xface_centred,
                             int y_face_centred, int halo_width, int field_set, int slot_start) {

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
    hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
      auto field = src_grid.access<R>(h);
      auto top_snd_buffer = top_snd_buffer_buffer.access<W>(h);
      hydro2d::par_ranged<class CONCAT_LINE(hydro2d_pack_edge_top)>(h, {x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [=](id<1> j) {
        size_t buf_pos = slot_start + k + j.get(0) * halo_width;
        top_snd_buffer[buf_pos] = field[j.get(0)][y_hi + 1 - k];
      });
    });
  }
}

void hydro2d_unpack_edge_top(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &src_grid,
                               hydro2d::DeviceArray1D<double> &top_rcv_buffer_buffer, int cell_centred, int node_centred, int xface_centred,
                               int y_face_centred, int halo_width, int field_set, int slot_start) {

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

    hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
      auto field = src_grid.access<W>(h);
      auto top_rcv_buffer = top_rcv_buffer_buffer.access<R>(h);
      hydro2d::par_ranged<class CONCAT_LINE(hydro2d_unpack_edge_top)>(h, {x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [=](id<1> j) {
        size_t buf_pos = slot_start + k + j.get(0) * halo_width;
        field[j.get(0)][y_hi + y_extra + k + 2] = top_rcv_buffer[buf_pos];
      });
    });
  }
}

void hydro2d_pack_edge_bottom(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi,
                                hydro2d::DeviceArray2D<double> &src_grid, hydro2d::DeviceArray1D<double> &bottom_snd_buffer_buffer, int cell_centred,
                                int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {

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

    hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
      auto field = src_grid.access<R>(h);
      auto bottom_snd_buffer = bottom_snd_buffer_buffer.access<W>(h);
      hydro2d::par_ranged<class CONCAT_LINE(hydro2d_pack_edge_bottom)>(h, {x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [=](id<1> j) {
        size_t buf_pos = slot_start + k + j.get(0) * halo_width;
        bottom_snd_buffer[buf_pos] = field[j.get(0)][y_lo + y_extra - 1 + k + 2];
      });
    });
  }
}

void hydro2d_unpack_edge_bottom(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi,
                                  hydro2d::DeviceArray2D<double> &src_grid, hydro2d::DeviceArray1D<double> &bottom_rcv_buffer_buffer, int cell_centred,
                                  int node_centred, int xface_centred, int y_face_centred, int halo_width, int field_set, int slot_start) {

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
    hydro2d::execute(sim.context.exec_ctx, [&](handler &h) {
      auto field = src_grid.access<W>(h);
      auto bottom_rcv_buffer = bottom_rcv_buffer_buffer.access<R>(h);
      hydro2d::par_ranged<class CONCAT_LINE(hydro2d_unpack_edge_top)>(h, {x_lo - halo_width + 1, x_hi + x_extra + halo_width + 2}, [=](id<1> j) {
        size_t buf_pos = slot_start + k + j.get(0) * halo_width;
        field[j.get(0)][y_lo - k] = bottom_rcv_buffer[buf_pos];
      });
    });
  }
}
