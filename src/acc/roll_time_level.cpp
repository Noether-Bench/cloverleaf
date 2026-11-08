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

#include "roll_time_level.h"
#include "wall_seconds.h"

//  @brief Fortran reset_time field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of cycle_index filed data to the begining of
//  cycle_index data, ready for the next compute_dt.
void roll_time_level_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field) {

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  const int cell_row_pitch = field.cell_row_pitch;
  double *density0 = field.density0.data;
  double *density1 = field.density1.data;
  double *energy0 = field.energy0.data;
  double *spec_energy1 = field.spec_energy1.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
  present(density0[ : field.density0.N()], density1[ : field.density1.N()],            \
          energy0[ : field.energy0.N()], spec_energy1[ : field.spec_energy1.N()])
  for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
      density0[i + j * cell_row_pitch] = density1[i + j * cell_row_pitch];
      energy0[i + j * cell_row_pitch] = spec_energy1[i + j * cell_row_pitch];
    }
  }

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  const int node_row_pitch = field.node_row_pitch;
  double *velx0 = field.velx0.data;
  double *velx1 = field.velx1.data;
  double *vely0 = field.vely0.data;
  double *vely1 = field.vely1.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
  present(velx0[ : field.velx0.N()], velx1[ : field.velx1.N()],                        \
          vely0[ : field.vely0.N()], vely1[ : field.vely1.N()])
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      velx0[i + j * node_row_pitch] = velx1[i + j * node_row_pitch];
      vely0[i + j * node_row_pitch] = vely1[i + j * node_row_pitch];
    }
  }
}

//  @brief Reset field solver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset_time kernel.
void roll_time_level(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    roll_time_level_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.reset_time += wall_seconds() - phase_start;
}
