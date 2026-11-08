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

#include "PdV.h"
#include "messaging.h"
#include "equation_of_state.h"
#include "diagnostics.h"
#include "rollback_state.h"
#include "wall_seconds.h"
#include "refresh_ghost_cells.h"
#include <cmath>

//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in initial_energy and initial_density in a cell using the
//  change on cell cell_volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void compression_work_kernel(bool use_target, bool predictor_pass, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field) {

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;
  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi

  if (predictor_pass) {

    double *xface_area = field.xface_area.data;

    double *yface_area = field.yface_area.data;
    double *cell_volume = field.cell_volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *velx0 = field.velx0.data;
    double *vely0 = field.vely0.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)         \
    present(xface_area[ : field.xface_area.N()], yface_area[ : field.yface_area.N()], cell_volume[ : field.cell_volume.N()], \
            density0[ : field.density0.N()], density1[ : field.density1.N()],                  \
            energy0[ : field.energy0.N()], spec_energy1[ : field.spec_energy1.N()],                      \
            cell_pressure[ : field.cell_pressure.N()], shock_damping[ : field.shock_damping.N()],                \
            velx0[ : field.velx0.N()], vely0[: field.vely0.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double west_face_flux = (xface_area[i + j * xflux_row_pitch] * (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                                            velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch])) *
                           0.25 * dt * 0.5;
        double east_face_flux = (xface_area[(i + 1) + (j + 0) * xflux_row_pitch] *
                             (velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch] +
                              velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch])) *
                            0.25 * dt * 0.5;
        double south_face_flux = (yface_area[i + j * yface_row_stride] * (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                                              vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch])) *
                             0.25 * dt * 0.5;
        double north_face_flux = (yface_area[(i + 0) + (j + 1) * yface_row_stride] *
                           (vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch] +
                            vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch])) *
                          0.25 * dt * 0.5;
        double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
        double volume_ratio_s = cell_volume[i + j * cell_row_pitch] / (cell_volume[i + j * cell_row_pitch] + net_face_flux);
        double inv_cell_volume = 1.0 / cell_volume[i + j * cell_row_pitch];
        double internal_energy_delta = (cell_pressure[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch] +
                                shock_damping[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch]) *
                               net_face_flux * inv_cell_volume;
        spec_energy1[i + j * cell_row_pitch] = energy0[i + j * cell_row_pitch] - internal_energy_delta;
        density1[i + j * cell_row_pitch] = density0[i + j * cell_row_pitch] * volume_ratio_s;
      }
    }

  } else {

    double *xface_area = field.xface_area.data;
    double *yface_area = field.yface_area.data;
    double *cell_volume = field.cell_volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *velx0 = field.velx0.data;
    double *velx1 = field.velx1.data;
    double *vely0 = field.vely0.data;
    double *vely1 = field.vely1.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)         \
    present(xface_area[ : field.xface_area.N()], yface_area[ : field.yface_area.N()], cell_volume[ : field.cell_volume.N()], \
            density0[ : field.density0.N()], density1[ : field.density1.N()],                  \
            energy0[ : field.energy0.N()], spec_energy1[ : field.spec_energy1.N()],                      \
            cell_pressure[ : field.cell_pressure.N()], shock_damping[ : field.shock_damping.N()],                \
            velx0[ : field.velx0.N()], velx1[ : field.velx1.N()],                              \
            vely0[: field.vely0.N()], vely1[ : field.vely1.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double west_face_flux = (xface_area[i + j * xflux_row_pitch] * (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                                            velx1[i + j * node_row_pitch] + velx1[(i + 0) + (j + 1) * node_row_pitch])) *
                           0.25 * dt;
        double east_face_flux = (xface_area[(i + 1) + (j + 0) * xflux_row_pitch] *
                             (velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch] +
                              velx1[(i + 1) + (j + 0) * node_row_pitch] + velx1[(i + 1) + (j + 1) * node_row_pitch])) *
                            0.25 * dt;
        double south_face_flux = (yface_area[i + j * yface_row_stride] * (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                                              vely1[i + j * node_row_pitch] + vely1[(i + 1) + (j + 0) * node_row_pitch])) *
                             0.25 * dt;
        double north_face_flux = (yface_area[(i + 0) + (j + 1) * yface_row_stride] *
                           (vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch] +
                            vely1[(i + 0) + (j + 1) * node_row_pitch] + vely1[(i + 1) + (j + 1) * node_row_pitch])) *
                          0.25 * dt;
        double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
        double volume_ratio_s = cell_volume[i + j * cell_row_pitch] / (cell_volume[i + j * cell_row_pitch] + net_face_flux);
        double inv_cell_volume = 1.0 / cell_volume[i + j * cell_row_pitch];
        double internal_energy_delta = (cell_pressure[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch] +
                                shock_damping[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch]) *
                               net_face_flux * inv_cell_volume;
        spec_energy1[i + j * cell_row_pitch] = energy0[i + j * cell_row_pitch] - internal_energy_delta;
        density1[i + j * cell_row_pitch] = density0[i + j * cell_row_pitch] * volume_ratio_s;
      }
    }
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(sim_state &globals, bool predictor_pass) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  globals.error_status = 0;

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    compression_work_kernel(globals.context.use_target, predictor_pass, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  hydro2d_max_error(globals.error_status);
  if (globals.timing_enabled) globals.kernel_timings.PdV += wall_seconds() - phase_start;

  if (globals.error_status == 1) {
    fatal_message((char *)"PdV", (char *)"error in PdV");
  }

  if (predictor_pass) {
    if (globals.timing_enabled) phase_start = wall_seconds();
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      equation_of_state(globals, tile_id, true);
    }

    if (globals.timing_enabled) globals.kernel_timings.equation_of_state += wall_seconds() - phase_start;

    int exchange_mask[FIELD_COUNT];
    for (int &field : exchange_mask)
      field = 0;
    exchange_mask[fld_pressure] = 1;
    refresh_ghost_cells(globals, exchange_mask, 1);
  }

  if (predictor_pass) {
    if (globals.timing_enabled) phase_start = wall_seconds();
    rollback_state(globals);
    if (globals.timing_enabled) globals.kernel_timings.rollback_state += wall_seconds() - phase_start;
  }
}
