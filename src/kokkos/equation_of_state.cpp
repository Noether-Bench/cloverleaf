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

//  @brief Fortran ideal gas kernel.
//  @author Wayne Gaudin
//  @details Calculates the cell_pressure and sound speed for the mesh subdomain using
//  the ideal gas equation of region_id, with a fixed gamma of 1.4.
void equation_of_state_kernel(int x_lo, int x_hi, int y_lo, int y_hi, Kokkos::View<double **> &initial_density, Kokkos::View<double **> &initial_energy,
                      Kokkos::View<double **> &cell_pressure, Kokkos::View<double **> &wave_speed) {

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});

  Kokkos::parallel_for(
      "equation_of_state", policy, KOKKOS_LAMBDA(const int j, const int k) {
        double v = 1.0 / initial_density(j, k);
        cell_pressure(j, k) = (1.4 - 1.0) * initial_density(j, k) * initial_energy(j, k);
        double dp_de = (1.4 - 1.0) * initial_density(j, k);
        double dp_dv = -initial_density(j, k) * cell_pressure(j, k);
        double cs_squared = v * v * (cell_pressure(j, k) * dp_de - dp_dv);
        wave_speed(j, k) = sqrt(cs_squared);
      });
}

//  @brief Ideal gas kernel solver
//  @author Wayne Gaudin
//  @details Invokes the user specified kernel for the ideal gas equation of
//  region_id using the specified time level data.

void equation_of_state(sim_state &globals, const int tile_id, bool predictor_pass) {

  if (!predictor_pass) {
    equation_of_state_kernel(globals.subdomain.tile_list[tile_id].meta.tile_xlo, globals.subdomain.tile_list[tile_id].meta.blk_xmax, globals.subdomain.tile_list[tile_id].meta.blk_ymin,
                     globals.subdomain.tile_list[tile_id].meta.blk_ymax, globals.subdomain.tile_list[tile_id].field.density0.host_vals,
                     globals.subdomain.tile_list[tile_id].field.energy0.host_vals, globals.subdomain.tile_list[tile_id].field.cell_pressure.host_vals,
                     globals.subdomain.tile_list[tile_id].field.wave_speed.host_vals);
  } else {
    equation_of_state_kernel(globals.subdomain.tile_list[tile_id].meta.tile_xlo, globals.subdomain.tile_list[tile_id].meta.blk_xmax, globals.subdomain.tile_list[tile_id].meta.blk_ymin,
                     globals.subdomain.tile_list[tile_id].meta.blk_ymax, globals.subdomain.tile_list[tile_id].field.density1.host_vals,
                     globals.subdomain.tile_list[tile_id].field.spec_energy1.host_vals, globals.subdomain.tile_list[tile_id].field.cell_pressure.host_vals,
                     globals.subdomain.tile_list[tile_id].field.wave_speed.host_vals);
  }
}
