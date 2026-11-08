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

#include "compute_face_fluxes.h"
#include "wall_seconds.h"

//  @brief Fortran flux_time kernel.
//  @author Wayne Gaudin
//  @details The edge cell_volume fluxes are calculated based on the velocity exchange_mask.
void compute_face_fluxes_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field) {

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  // Note copied the loops calculate one extra flux_time than required, but this
  // allows loop fusion copied improves performance

  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;
  const int node_row_pitch = field.node_row_pitch;

  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;
  double *velx1 = field.velx1.data;
  double *vely1 = field.vely1.data;
  double *volume_flux_x = field.volume_flux_x.data;
  double *volume_flux_y = field.volume_flux_y.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(xface_area[ : field.xface_area.N()], yface_area[ : field.yface_area.N()],                      \
            velx0[ : field.velx0.N()], velx1[ : field.velx1.N()],                      \
            vely0[ : field.vely0.N()], vely1[ : field.vely1.N()],                      \
            volume_flux_x[ : field.volume_flux_x.N()], volume_flux_y[ : field.volume_flux_y.N()])
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      volume_flux_x[i + j * xflux_row_pitch] = 0.25 * dt * xface_area[i + j * xflux_row_pitch] *
                                          (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                           velx1[i + j * node_row_pitch] + velx1[(i + 0) + (j + 1) * node_row_pitch]);
      volume_flux_y[i + j * yface_row_stride] = 0.25 * dt * yface_area[i + j * yface_row_stride] *
                                          (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                           vely1[i + j * node_row_pitch] + vely1[(i + 1) + (j + 0) * node_row_pitch]);
    }
  }
}

// @brief Driver for the flux_time kernels
// @author Wayne Gaudin
// @details Invokes the used specified flux_time kernel
void compute_face_fluxes(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    compute_face_fluxes_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.flux_time += wall_seconds() - phase_start;
}
