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
void advance_velocity_kernel(int x_lo, int x_hi, int y_lo, int y_hi, double dt, Kokkos::View<double **> &xface_area,
                       Kokkos::View<double **> &yface_area, Kokkos::View<double **> &cell_volume, Kokkos::View<double **> &density0,
                       Kokkos::View<double **> &cell_pressure, Kokkos::View<double **> &shock_damping, Kokkos::View<double **> &velx0,
                       Kokkos::View<double **> &vely0, Kokkos::View<double **> &velx1, Kokkos::View<double **> &vely1) {

  double half_step = 0.5 * dt;

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 1 + 2, y_hi + 1 + 2});
  Kokkos::parallel_for(
      "advance_velocity", policy, KOKKOS_LAMBDA(const int j, const int k) {
        double dt_over_mass_s = half_step / ((density0(j - 1, k - 1) * cell_volume(j - 1, k - 1) + density0(j, k - 1) * cell_volume(j, k - 1) +
                                         density0(j, k) * cell_volume(j, k) + density0(j - 1, k) * cell_volume(j - 1, k)) *
                                        0.25);

        velx1(j, k) = velx0(j, k) - dt_over_mass_s * (xface_area(j, k) * (cell_pressure(j, k) - cell_pressure(j - 1, k)) +
                                                    xface_area(j, k - 1) * (cell_pressure(j, k - 1) - cell_pressure(j - 1, k - 1)));
        vely1(j, k) = vely0(j, k) - dt_over_mass_s * (yface_area(j, k) * (cell_pressure(j, k) - cell_pressure(j, k - 1)) +
                                                    yface_area(j - 1, k) * (cell_pressure(j - 1, k) - cell_pressure(j - 1, k - 1)));
        velx1(j, k) = velx1(j, k) - dt_over_mass_s * (xface_area(j, k) * (shock_damping(j, k) - shock_damping(j - 1, k)) +
                                                    xface_area(j, k - 1) * (shock_damping(j, k - 1) - shock_damping(j - 1, k - 1)));
        vely1(j, k) = vely1(j, k) - dt_over_mass_s * (yface_area(j, k) * (shock_damping(j, k) - shock_damping(j, k - 1)) +
                                                    yface_area(j - 1, k) * (shock_damping(j - 1, k) - shock_damping(j - 1, k - 1)));
      });
}

//  @brief Driver for the accel_time kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void advance_velocity(sim_state &globals) {

  double phase_start{};
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    advance_velocity_kernel(globals.subdomain.tile_list[tile_id].meta.tile_xlo, globals.subdomain.tile_list[tile_id].meta.blk_xmax, globals.subdomain.tile_list[tile_id].meta.blk_ymin,
                      globals.subdomain.tile_list[tile_id].meta.blk_ymax, globals.dt, globals.subdomain.tile_list[tile_id].field.xface_area.host_vals,
                      globals.subdomain.tile_list[tile_id].field.yface_area.host_vals, globals.subdomain.tile_list[tile_id].field.cell_volume.host_vals,
                      globals.subdomain.tile_list[tile_id].field.density0.host_vals, globals.subdomain.tile_list[tile_id].field.cell_pressure.host_vals,
                      globals.subdomain.tile_list[tile_id].field.shock_damping.host_vals, globals.subdomain.tile_list[tile_id].field.velx0.host_vals,
                      globals.subdomain.tile_list[tile_id].field.vely0.host_vals, globals.subdomain.tile_list[tile_id].field.velx1.host_vals,
                      globals.subdomain.tile_list[tile_id].field.vely1.host_vals);
  }

  if (globals.timing_enabled) globals.kernel_timings.accel_time += wall_seconds() - phase_start;
}
