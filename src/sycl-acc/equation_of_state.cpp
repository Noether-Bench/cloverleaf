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

#include "equation_of_state.h"
#include "context.h"

#define IDX(field_buf, x, y) field_buf[idx[(x)]][idx[(y)]]

#define SWP(x, y) (x), (y)

int N = 0;

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the cell_pressure and sound speed for the mesh subdomain using
//  the ideal gas equation of region_id, with a fixed gamma of 1.4.
void equation_of_state_kernel(handler &h, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::Accessor<double, 2, R>::Type initial_density,
                      hydro2d::Accessor<double, 2, R>::Type initial_energy, hydro2d::Accessor<double, 2, RW>::Type cell_pressure,
                      hydro2d::Accessor<double, 2, W>::Type wave_speed) {

  // std::cout <<" equation_of_state(" << x_lo+1 << ","<< y_lo+1<< ","<< x_hi+2<< ","<< y_hi +2  << ")" << std::endl;
  //  DO k=y_lo,y_hi
  //    DO j=x_lo,x_hi

  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});

  hydro2d::par_ranged<class equation_of_state>(h, {x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=](id<2> idx) {
    double v = 1.0 / initial_density[idx];
    cell_pressure[idx] = (1.4 - 1.0) * initial_density[idx] * initial_energy[idx];
    double dp_de = (1.4 - 1.0) * initial_density[idx];
    double dp_dv = -initial_density[idx] * cell_pressure[idx];
    double cs_squared = v * v * (cell_pressure[idx] * dp_de - dp_dv);
    wave_speed[idx] = sycl::sqrt(cs_squared);
  });
}

//  @brief Ideal gas kernel solver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  region_id using the specified time level data.

void equation_of_state(sim_state &globals, const int tile_id, bool predictor_pass) {

  tile_block &t = globals.subdomain.tile_list[tile_id];

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    if (!predictor_pass) {
      equation_of_state_kernel(h, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0.access<R>(h),
                       t.field.energy0.access<R>(h), t.field.cell_pressure.access<RW>(h), t.field.wave_speed.access<W>(h));
    } else {
      equation_of_state_kernel(h, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1.access<R>(h),
                       t.field.spec_energy1.access<R>(h), t.field.cell_pressure.access<RW>(h), t.field.wave_speed.access<W>(h));
    }
  });
}
