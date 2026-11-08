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
void advance_velocity_kernel(handler &h, int x_lo, int x_hi, int y_lo, int y_hi, double dt, //
                       hydro2d::Accessor<double, 2, R>::Type xface_area, hydro2d::Accessor<double, 2, R>::Type yface_area,
                       hydro2d::Accessor<double, 2, R>::Type cell_volume, hydro2d::Accessor<double, 2, R>::Type density0,
                       hydro2d::Accessor<double, 2, R>::Type cell_pressure, hydro2d::Accessor<double, 2, R>::Type shock_damping,
                       hydro2d::Accessor<double, 2, R>::Type velx0, hydro2d::Accessor<double, 2, R>::Type vely0,
                       hydro2d::Accessor<double, 2, RW>::Type velx1, hydro2d::Accessor<double, 2, RW>::Type vely1) {

  double half_step = 0.5 * dt;

  // DO k=y_lo,y_hi+1
  //   DO j=x_lo,x_hi+1
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1},
  //	                                               {x_hi + 1 + 2, y_hi + 1 + 2});

  hydro2d::par_ranged<class advance_velocity>(h, {x_lo + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
    double dt_over_mass_s =
        half_step / ((density0[hydro2d::offset(idx, -1, -1)] * cell_volume[hydro2d::offset(idx, -1, -1)] +
                   density0[hydro2d::offset(idx, -1, 0)] * cell_volume[hydro2d::offset(idx, -1, 0)] + density0[idx] * cell_volume[idx] +
                   density0[hydro2d::offset(idx, 0, -1)] * cell_volume[hydro2d::offset(idx, 0, -1)]) *
                  0.25);

    velx1[idx] = velx0[idx] - dt_over_mass_s * (xface_area[idx] * (cell_pressure[idx] - cell_pressure[hydro2d::offset(idx, -1, 0)]) +
                                              xface_area[hydro2d::offset(idx, 0, -1)] *
                                                  (cell_pressure[hydro2d::offset(idx, 0, -1)] - cell_pressure[hydro2d::offset(idx, -1, -1)]));
    vely1[idx] = vely0[idx] - dt_over_mass_s * (yface_area[idx] * (cell_pressure[idx] - cell_pressure[hydro2d::offset(idx, 0, -1)]) +
                                              yface_area[hydro2d::offset(idx, -1, 0)] *
                                                  (cell_pressure[hydro2d::offset(idx, -1, 0)] - cell_pressure[hydro2d::offset(idx, -1, -1)]));
    velx1[idx] = velx1[idx] - dt_over_mass_s * (xface_area[idx] * (shock_damping[idx] - shock_damping[hydro2d::offset(idx, -1, 0)]) +
                                              xface_area[hydro2d::offset(idx, 0, -1)] *
                                                  (shock_damping[hydro2d::offset(idx, 0, -1)] - shock_damping[hydro2d::offset(idx, -1, -1)]));
    vely1[idx] = vely1[idx] - dt_over_mass_s * (yface_area[idx] * (shock_damping[idx] - shock_damping[hydro2d::offset(idx, 0, -1)]) +
                                              yface_area[hydro2d::offset(idx, -1, 0)] *
                                                  (shock_damping[hydro2d::offset(idx, -1, 0)] - shock_damping[hydro2d::offset(idx, -1, -1)]));
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

    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      advance_velocity_kernel(h, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt,
                        t.field.xface_area.access<R>(h),     //
                        t.field.yface_area.access<R>(h),     //
                        t.field.cell_volume.access<R>(h),    //
                        t.field.density0.access<R>(h),  //
                        t.field.cell_pressure.access<R>(h),  //
                        t.field.shock_damping.access<R>(h), //
                        t.field.velx0.access<R>(h),     //
                        t.field.vely0.access<R>(h),     //
                        t.field.velx1.access<RW>(h),    //
                        t.field.vely1.access<RW>(h)     //
      );
    });
  }

  if (globals.timing_enabled) globals.kernel_timings.accel_time += wall_seconds() - phase_start;
}
