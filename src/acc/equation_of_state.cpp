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
#include <cmath>

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the cell_pressure and sound speed for the mesh subdomain using
//  the ideal gas equation of region_id, with a fixed gamma of 1.4.
void equation_of_state_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field,
                      hydro2d::DeviceArray2D<double> &density_in, hydro2d::DeviceArray2D<double> &energy_in) {

  // std::cout <<" equation_of_state(" << x_lo+1 << ","<< y_lo+1<< ","<< x_hi+2<< ","<< y_hi +2  << ")" << std::endl;
  //  DO k=y_lo,y_hi
  //    DO j=x_lo,x_hi

  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});

  const size_t cell_row_pitch = field.cell_row_pitch;

  double *initial_density = density_in.data;
  double *initial_energy = energy_in.data;
  double *cell_pressure = field.cell_pressure.data;
  double *wave_speed = field.wave_speed.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
  present(initial_density[ : density_in.N()], initial_energy[ : energy_in.N()],                \
          cell_pressure[ : field.cell_pressure.N()], wave_speed[ : field.wave_speed.N()])
  for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
      double v = 1.0 / initial_density[i + j * cell_row_pitch];
      cell_pressure[i + j * cell_row_pitch] = (1.4 - 1.0) * initial_density[i + j * cell_row_pitch] * initial_energy[i + j * cell_row_pitch];
      double dp_de = (1.4 - 1.0) * initial_density[i + j * cell_row_pitch];
      double dp_dv = -initial_density[i + j * cell_row_pitch] * cell_pressure[i + j * cell_row_pitch];
      double cs_squared = v * v * (cell_pressure[i + j * cell_row_pitch] * dp_de - dp_dv);
      wave_speed[i + j * cell_row_pitch] = std::sqrt(cs_squared);
    }
  };
}

//  @brief Ideal gas kernel solver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  region_id using the specified time level data.

void equation_of_state(sim_state &globals, const int tile_id, bool predictor_pass) {

  tile_block &t = globals.subdomain.tile_list[tile_id];

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  if (!predictor_pass) {
    equation_of_state_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field, t.field.density0,
                     t.field.energy0);
  } else {
    equation_of_state_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field, t.field.density1,
                     t.field.spec_energy1);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif
}
