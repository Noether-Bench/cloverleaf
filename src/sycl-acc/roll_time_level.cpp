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
#include "context.h"
#include "wall_seconds.h"

//  @brief Fortran reset_time field kernel.
//  @author Wayne Gaudin
//  @details Copies all of the final end of cycle_index filed data to the begining of
//  cycle_index data, ready for the next compute_dt.
void roll_time_level_kernel(exec_ctx &q, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &local_density0_buf,
                        hydro2d::DeviceArray2D<double> &local_density1_buf, hydro2d::DeviceArray2D<double> &local_energy0_buf,
                        hydro2d::DeviceArray2D<double> &local_energy1_buf, hydro2d::DeviceArray2D<double> &local_velx0_buf,
                        hydro2d::DeviceArray2D<double> &local_velx1_buf, hydro2d::DeviceArray2D<double> &local_vely0_buf,
                        hydro2d::DeviceArray2D<double> &local_vely1_buf) {

  hydro2d::execute(q, [&](handler &h) {
    auto density0 = local_density0_buf.access<W>(h);
    auto density1 = local_density1_buf.access<R>(h);
    auto energy0 = local_energy0_buf.access<W>(h);
    auto spec_energy1 = local_energy1_buf.access<R>(h);
    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi
    hydro2d::par_ranged<class reset_field_1>(h, {x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=](id<2> idx) {
      density0[idx] = density1[idx];
      energy0[idx] = spec_energy1[idx];
    });
  });

  hydro2d::execute(q, [&](handler &h) {
    auto velx1 = local_velx1_buf.access<R>(h);
    auto vely0 = local_vely0_buf.access<W>(h);
    auto vely1 = local_vely1_buf.access<R>(h);
    auto velx0 = local_velx0_buf.access<W>(h);
    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1
    hydro2d::par_ranged<class reset_field_2>(h, {x_lo + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
      velx0[idx] = velx1[idx];
      vely0[idx] = vely1[idx];
    });
  });
}

//  @brief Reset field solver
//  @author Wayne Gaudin
//  @details Invokes the user specified field reset_time kernel.
void roll_time_level(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    roll_time_level_kernel(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                       t.field.density0, //
                       t.field.density1, //
                       t.field.energy0,  //
                       t.field.spec_energy1,  //
                       t.field.velx0,    //
                       t.field.velx1,    //
                       t.field.vely0,    //
                       t.field.vely1     //
    );
  }

  if (globals.timing_enabled) globals.kernel_timings.reset_time += wall_seconds() - phase_start;
}
