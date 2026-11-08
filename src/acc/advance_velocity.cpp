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

#include "advance_velocity.h"
#include "wall_seconds.h"

// @brief Fortran accel_time kernel
// @author Wayne Gaudin
// @details The cell_pressure and shock_damping gradients are used to update the
// velocity field.
void advance_velocity_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field) {

  double half_step = 0.5 * dt;

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1},
  //	                                               {x_hi + 1 + 2, y_hi + 1 + 2});

  // for(int j = )

  const int xface_area_nx = field.xflux_row_pitch;
  const int yface_area_nx = field.yface_row_stride;
  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;

  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;
  double *cell_volume = field.cell_volume.data;
  double *density0 = field.density0.data;
  double *cell_pressure = field.cell_pressure.data;
  double *shock_damping = field.shock_damping.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;
  double *velx1 = field.velx1.data;
  double *vely1 = field.vely1.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)        \
    present(xface_area[ : field.xface_area.N()], yface_area[ : field.yface_area.N()],                             \
            cell_volume[ : field.cell_volume.N()], density0[ : field.density0.N()],                     \
            cell_pressure[ : field.cell_pressure.N()], shock_damping[ : field.shock_damping.N()],               \
            velx0[ : field.velx0.N()], velx1[ : field.velx1.N()],                             \
            vely0[: field.vely0.N()], vely1[ : field.vely1.N()])
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      double dt_over_mass_s = half_step / ((density0[(i - 1) + (j - 1) * cell_row_pitch] * cell_volume[(i - 1) + (j - 1) * cell_row_pitch] +
                                       density0[(i - 1) + (j + 0) * cell_row_pitch] * cell_volume[(i - 1) + (j + 0) * cell_row_pitch] +
                                       density0[i + j * cell_row_pitch] * cell_volume[i + j * cell_row_pitch] +
                                       density0[(i + 0) + (j - 1) * cell_row_pitch] * cell_volume[(i + 0) + (j - 1) * cell_row_pitch]) *
                                      0.25);
      velx1[i + j * node_row_pitch] =
          velx0[i + j * node_row_pitch] -
          dt_over_mass_s * (xface_area[i + j * xface_area_nx] * (cell_pressure[i + j * cell_row_pitch] - cell_pressure[(i - 1) + (j + 0) * cell_row_pitch]) +
                          xface_area[(i + 0) + (j - 1) * xface_area_nx] *
                              (cell_pressure[(i + 0) + (j - 1) * cell_row_pitch] - cell_pressure[(i - 1) + (j - 1) * cell_row_pitch]));
      vely1[i + j * node_row_pitch] =
          vely0[i + j * node_row_pitch] -
          dt_over_mass_s * (yface_area[i + j * yface_area_nx] * (cell_pressure[i + j * cell_row_pitch] - cell_pressure[(i + 0) + (j - 1) * cell_row_pitch]) +
                          yface_area[(i - 1) + (j + 0) * yface_area_nx] *
                              (cell_pressure[(i - 1) + (j + 0) * cell_row_pitch] - cell_pressure[(i - 1) + (j - 1) * cell_row_pitch]));
      velx1[i + j * node_row_pitch] =
          velx1[i + j * node_row_pitch] -
          dt_over_mass_s * (xface_area[i + j * xface_area_nx] * (shock_damping[i + j * cell_row_pitch] - shock_damping[(i - 1) + (j + 0) * cell_row_pitch]) +
                          xface_area[(i + 0) + (j - 1) * xface_area_nx] *
                              (shock_damping[(i + 0) + (j - 1) * cell_row_pitch] - shock_damping[(i - 1) + (j - 1) * cell_row_pitch]));
      vely1[i + j * node_row_pitch] =
          vely1[i + j * node_row_pitch] -
          dt_over_mass_s * (yface_area[i + j * yface_area_nx] * (shock_damping[i + j * cell_row_pitch] - shock_damping[(i + 0) + (j - 1) * cell_row_pitch]) +
                          yface_area[(i - 1) + (j + 0) * yface_area_nx] *
                              (shock_damping[(i - 1) + (j + 0) * cell_row_pitch] - shock_damping[(i - 1) + (j - 1) * cell_row_pitch]));
    }
  }
}

//  @brief Driver for the accel_time kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void advance_velocity(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];

    advance_velocity_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.accel_time += wall_seconds() - phase_start;
}
