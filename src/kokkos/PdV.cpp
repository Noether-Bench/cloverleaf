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

//  @brief Fortran PdV kernel.
//  @author Wayne Gaudin
//  @details Calculates the change in initial_energy and initial_density in a cell using the
//  change on cell cell_volume due to the velocity gradients in a cell. The time
//  level of the velocity data depends on whether it is invoked as the
//  predictor or corrector.
void compression_work_kernel(bool predictor_pass, int x_lo, int x_hi, int y_lo, int y_hi, double dt, Kokkos::View<double **> &xface_area,
                Kokkos::View<double **> &yface_area, Kokkos::View<double **> &cell_volume, Kokkos::View<double **> &density0,
                Kokkos::View<double **> &density1, Kokkos::View<double **> &energy0, Kokkos::View<double **> &spec_energy1,
                Kokkos::View<double **> &cell_pressure, Kokkos::View<double **> &shock_damping, Kokkos::View<double **> &velx0,
                Kokkos::View<double **> &velx1, Kokkos::View<double **> &vely0, Kokkos::View<double **> &vely1,
                Kokkos::View<double **> &volume_change) {

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});

  if (predictor_pass) {

    Kokkos::parallel_for(
        "PdV predictor_pass=true", policy, KOKKOS_LAMBDA(const int j, const int k) {
          double west_face_flux = (xface_area(j, k) * (velx0(j, k) + velx0(j, k + 1) + velx0(j, k) + velx0(j, k + 1))) * 0.25 * dt * 0.5;
          double east_face_flux =
              (xface_area(j + 1, k) * (velx0(j + 1, k) + velx0(j + 1, k + 1) + velx0(j + 1, k) + velx0(j + 1, k + 1))) * 0.25 * dt * 0.5;
          double south_face_flux = (yface_area(j, k) * (vely0(j, k) + vely0(j + 1, k) + vely0(j, k) + vely0(j + 1, k))) * 0.25 * dt * 0.5;
          double north_face_flux =
              (yface_area(j, k + 1) * (vely0(j, k + 1) + vely0(j + 1, k + 1) + vely0(j, k + 1) + vely0(j + 1, k + 1))) * 0.25 * dt * 0.5;
          double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
          double volume_ratio_s = cell_volume(j, k) / (cell_volume(j, k) + net_face_flux);
          double inv_cell_volume = 1.0 / cell_volume(j, k);
          double internal_energy_delta = (cell_pressure(j, k) / density0(j, k) + shock_damping(j, k) / density0(j, k)) * net_face_flux * inv_cell_volume;
          spec_energy1(j, k) = energy0(j, k) - internal_energy_delta;
          density1(j, k) = density0(j, k) * volume_ratio_s;
        });

  } else {

    Kokkos::parallel_for(
        "PdV predictor_pass=false", policy, KOKKOS_LAMBDA(const int j, const int k) {
          double west_face_flux = (xface_area(j, k) * (velx0(j, k) + velx0(j, k + 1) + velx1(j, k) + velx1(j, k + 1))) * 0.25 * dt;
          double east_face_flux =
              (xface_area(j + 1, k) * (velx0(j + 1, k) + velx0(j + 1, k + 1) + velx1(j + 1, k) + velx1(j + 1, k + 1))) * 0.25 * dt;
          double south_face_flux = (yface_area(j, k) * (vely0(j, k) + vely0(j + 1, k) + vely1(j, k) + vely1(j + 1, k))) * 0.25 * dt;
          double north_face_flux = (yface_area(j, k + 1) * (vely0(j, k + 1) + vely0(j + 1, k + 1) + vely1(j, k + 1) + vely1(j + 1, k + 1))) * 0.25 * dt;
          double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
          double volume_ratio_s = cell_volume(j, k) / (cell_volume(j, k) + net_face_flux);
          double inv_cell_volume = 1.0 / cell_volume(j, k);
          double internal_energy_delta = (cell_pressure(j, k) / density0(j, k) + shock_damping(j, k) / density0(j, k)) * net_face_flux * inv_cell_volume;
          spec_energy1(j, k) = energy0(j, k) - internal_energy_delta;
          density1(j, k) = density0(j, k) * volume_ratio_s;
        });
  }
}

//  @brief Driver for the PdV update.
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the PdV update.
void PdV(sim_state &globals, bool predictor_pass) {

  double phase_start{};
  if (globals.timing_enabled) phase_start = wall_seconds();

  globals.error_status = 0;

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    compression_work_kernel(predictor_pass, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field.xface_area.host_vals, t.field.yface_area.host_vals,
               t.field.cell_volume.host_vals, t.field.density0.host_vals, t.field.density1.host_vals, t.field.energy0.host_vals, t.field.spec_energy1.host_vals,
               t.field.cell_pressure.host_vals, t.field.shock_damping.host_vals, t.field.velx0.host_vals, t.field.velx1.host_vals, t.field.vely0.host_vals,
               t.field.vely1.host_vals, t.field.scratch_buf1.host_vals);
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
    for (int i = 0; i < FIELD_COUNT; ++i)
      exchange_mask[i] = 0;
    exchange_mask[fld_pressure] = 1;
    refresh_ghost_cells(globals, exchange_mask, 1);
  }

  if (predictor_pass) {
    if (globals.timing_enabled) phase_start = wall_seconds();
    rollback_state(globals);
    if (globals.timing_enabled) globals.kernel_timings.rollback_state += wall_seconds() - phase_start;
  }
}
