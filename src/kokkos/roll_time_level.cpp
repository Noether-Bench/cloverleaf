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
void roll_time_level_kernel(int x_lo, int x_hi, int y_lo, int y_hi, Kokkos::View<double **> &density0, Kokkos::View<double **> &density1,
                        Kokkos::View<double **> &energy0, Kokkos::View<double **> &spec_energy1, Kokkos::View<double **> &velx0,
                        Kokkos::View<double **> &velx1, Kokkos::View<double **> &vely0, Kokkos::View<double **> &vely1) {

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy1({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});
  Kokkos::parallel_for(
      "reset_field_1", policy1, KOKKOS_LAMBDA(const int j, const int k) {
        density0(j, k) = density1(j, k);
        energy0(j, k) = spec_energy1(j, k);
      });

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy2({x_lo + 1, y_lo + 1}, {x_hi + 1 + 2, y_hi + 1 + 2});
  Kokkos::parallel_for(
      "reset_field_2", policy2, KOKKOS_LAMBDA(const int j, const int k) {
        velx0(j, k) = velx1(j, k);
        vely0(j, k) = vely1(j, k);
      });
}

//  @brief Reset field solver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset_time kernel.
void roll_time_level(sim_state &globals) {

  double phase_start{};
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    roll_time_level_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0.host_vals, t.field.density1.host_vals,
                       t.field.energy0.host_vals, t.field.spec_energy1.host_vals, t.field.velx0.host_vals, t.field.velx1.host_vals, t.field.vely0.host_vals,
                       t.field.vely1.host_vals);
  }

  if (globals.timing_enabled) globals.kernel_timings.reset_time += wall_seconds() - phase_start;
}
