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
#include "context.h"
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
void compression_work_kernel(sycl::exec_ctx &exec_ctx, bool predictor_pass, int x_lo, int x_hi, int y_lo, int y_hi, double dt, hydro2d::DeviceArray2D<double> &xface_area,
                hydro2d::DeviceArray2D<double> &yface_area, hydro2d::DeviceArray2D<double> &cell_volume, hydro2d::DeviceArray2D<double> &density0,
                hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &spec_energy1,
                hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping, hydro2d::DeviceArray2D<double> &velx0,
                hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &vely1) {

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  hydro2d::Span2d policy(x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2);

  if (predictor_pass) {

    hydro2d::par_ranged2(exec_ctx, policy, [=](const int i, const int j) {
      double west_face_flux = (xface_area(i, j) * (velx0(i, j) + velx0(i + 0, j + 1) + velx0(i, j) + velx0(i + 0, j + 1))) * 0.25 * dt * 0.5;
      double east_face_flux =
          (xface_area(i + 1, j + 0) * (velx0(i + 1, j + 0) + velx0(i + 1, j + 1) + velx0(i + 1, j + 0) + velx0(i + 1, j + 1))) * 0.25 * dt * 0.5;
      double south_face_flux = (yface_area(i, j) * (vely0(i, j) + vely0(i + 1, j + 0) + vely0(i, j) + vely0(i + 1, j + 0))) * 0.25 * dt * 0.5;
      double north_face_flux =
          (yface_area(i + 0, j + 1) * (vely0(i + 0, j + 1) + vely0(i + 1, j + 1) + vely0(i + 0, j + 1) + vely0(i + 1, j + 1))) * 0.25 * dt * 0.5;
      double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
      double volume_ratio_s = cell_volume(i, j) / (cell_volume(i, j) + net_face_flux);
      double inv_cell_volume = 1.0 / cell_volume(i, j);
      double internal_energy_delta = (cell_pressure(i, j) / density0(i, j) + shock_damping(i, j) / density0(i, j)) * net_face_flux * inv_cell_volume;
      spec_energy1(i, j) = energy0(i, j) - internal_energy_delta;
      density1(i, j) = density0(i, j) * volume_ratio_s;
    });

  } else {

    hydro2d::par_ranged2(exec_ctx, policy, [=](const int i, const int j) {
      double west_face_flux = (xface_area(i, j) * (velx0(i, j) + velx0(i + 0, j + 1) + velx1(i, j) + velx1(i + 0, j + 1))) * 0.25 * dt;
      double east_face_flux =
          (xface_area(i + 1, j + 0) * (velx0(i + 1, j + 0) + velx0(i + 1, j + 1) + velx1(i + 1, j + 0) + velx1(i + 1, j + 1))) * 0.25 * dt;
      double south_face_flux = (yface_area(i, j) * (vely0(i, j) + vely0(i + 1, j + 0) + vely1(i, j) + vely1(i + 1, j + 0))) * 0.25 * dt;
      double north_face_flux =
          (yface_area(i + 0, j + 1) * (vely0(i + 0, j + 1) + vely0(i + 1, j + 1) + vely1(i + 0, j + 1) + vely1(i + 1, j + 1))) * 0.25 * dt;
      double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
      double volume_ratio_s = cell_volume(i, j) / (cell_volume(i, j) + net_face_flux);
      double inv_cell_volume = 1.0 / cell_volume(i, j);
      double internal_energy_delta = (cell_pressure(i, j) / density0(i, j) + shock_damping(i, j) / density0(i, j)) * net_face_flux * inv_cell_volume;
      spec_energy1(i, j) = energy0(i, j) - internal_energy_delta;
      density1(i, j) = density0(i, j) * volume_ratio_s;
    });
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(sim_state &globals, bool predictor_pass) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  globals.error_status = 0;

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    compression_work_kernel(globals.context.exec_ctx, predictor_pass, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field.xface_area,
               t.field.yface_area, t.field.cell_volume, t.field.density0, t.field.density1, t.field.energy0, t.field.spec_energy1, t.field.cell_pressure,
               t.field.shock_damping, t.field.velx0, t.field.velx1, t.field.vely0, t.field.vely1);
  }

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
