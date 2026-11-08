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
#include "context.h"
#include "wall_seconds.h"

// @brief Fortran accel_time kernel
// @author Wayne Gaudin
// @details The cell_pressure and shock_damping gradients are used to update the
// velocity field.
void advance_velocity_kernel(int x_lo, int x_hi, int y_lo, int y_hi, double dt, hydro2d::DeviceArray2D<double> &xface_area,
                       hydro2d::DeviceArray2D<double> &yface_area, hydro2d::DeviceArray2D<double> &cell_volume, hydro2d::DeviceArray2D<double> &density0,
                       hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping, hydro2d::DeviceArray2D<double> &velx0,
                       hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1) {

  double half_step = 0.5 * dt;

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1},
  //	                                               {x_hi + 1 + 2, y_hi + 1 + 2});

  hydro2d::par_ranged2(Span2d{x_lo + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](const int i, const int j) {
    double dt_over_mass_s = half_step / ((density0(i - 1, j - 1) * cell_volume(i - 1, j - 1) + density0(i - 1, j + 0) * cell_volume(i - 1, j + 0) +
                                     density0(i, j) * cell_volume(i, j) + density0(i + 0, j - 1) * cell_volume(i + 0, j - 1)) *
                                    0.25);

    velx1(i, j) = velx0(i, j) - dt_over_mass_s * (xface_area(i, j) * (cell_pressure(i, j) - cell_pressure(i - 1, j + 0)) +
                                                xface_area(i + 0, j - 1) * (cell_pressure(i + 0, j - 1) - cell_pressure(i - 1, j - 1)));
    vely1(i, j) = vely0(i, j) - dt_over_mass_s * (yface_area(i, j) * (cell_pressure(i, j) - cell_pressure(i + 0, j - 1)) +
                                                yface_area(i - 1, j + 0) * (cell_pressure(i - 1, j + 0) - cell_pressure(i - 1, j - 1)));
    velx1(i, j) = velx1(i, j) - dt_over_mass_s * (xface_area(i, j) * (shock_damping(i, j) - shock_damping(i - 1, j + 0)) +
                                                xface_area(i + 0, j - 1) * (shock_damping(i + 0, j - 1) - shock_damping(i - 1, j - 1)));
    vely1(i, j) = vely1(i, j) - dt_over_mass_s * (yface_area(i, j) * (shock_damping(i, j) - shock_damping(i + 0, j - 1)) +
                                                yface_area(i - 1, j + 0) * (shock_damping(i - 1, j + 0) - shock_damping(i - 1, j - 1)));
  });
}

//  @brief Driver for the accel_time kernels
//  @author Wayne Gaudin
//  @details Calls user requested kernel
void advance_velocity(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];

    advance_velocity_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field.xface_area, t.field.yface_area, t.field.cell_volume,
                      t.field.density0, t.field.cell_pressure, t.field.shock_damping, t.field.velx0, t.field.vely0, t.field.velx1, t.field.vely1);
  }

  if (globals.timing_enabled) globals.kernel_timings.accel_time += wall_seconds() - phase_start;
}
