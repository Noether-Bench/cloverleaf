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

//  @brief Mesh subdomain generation solver
//  @author Wayne Gaudin
//  @details Invoked the users specified subdomain generator.
//  @brief Mesh subdomain generation solver
//  @author Wayne Gaudin
//  @details Invoked the users specified subdomain generator.

#include "populate_subdomain.h"
#include <cmath>

void populate_subdomain(const int tile_id, sim_state &globals) {

  // Need to copy the host array of region_id input data into a target_option array
  hydro2d::DeviceArray1D<double> region_density_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_energy_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_velx_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_vely_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_xlo_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_xhi_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_ylo_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_yhi_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_radius_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<int> region_shape_buf(globals.context, globals.settings.region_count);

  // Copy the data to the new views
  for (int region_id = 0; region_id < globals.settings.region_count; ++region_id) {
    region_density_buf[region_id] = globals.settings.regions[region_id].initial_density;
    region_energy_buf[region_id] = globals.settings.regions[region_id].initial_energy;
    region_velx_buf[region_id] = globals.settings.regions[region_id].velx;
    region_vely_buf[region_id] = globals.settings.regions[region_id].vely;
    region_xlo_buf[region_id] = globals.settings.regions[region_id].x_lower;
    region_xhi_buf[region_id] = globals.settings.regions[region_id].x_upper;
    region_ylo_buf[region_id] = globals.settings.regions[region_id].y_lower;
    region_yhi_buf[region_id] = globals.settings.regions[region_id].y_upper;
    region_radius_buf[region_id] = globals.settings.regions[region_id].circle_radius;
    region_shape_buf[region_id] = globals.settings.regions[region_id].shape_selector;
  }

  // Kokkos::deep_copy (TO, FROM)

  const int x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  const int x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  const int y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  const int y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  int x_extent = (x_hi + 2) - (x_lo - 2) + 1;
  int y_extent = (y_hi + 2) - (y_lo - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.

  field_set &field = globals.subdomain.tile_list[tile_id].field;

  const double background_energy = region_energy_buf[0];
  const double background_density = region_density_buf[0];
  const double background_velx = region_velx_buf[0];
  const double background_vely = region_vely_buf[0];

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;

  // State 1 is always the background region_id
  double *energy0 = field.energy0.data;
  double *density0 = field.density0.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(globals.context.use_target) \
    present(energy0[ : field.energy0.N()], density0[ : field.density0.N()],                            \
            velx0[ : field.velx0.N()], vely0[: field.vely0.N()])
  for (int j = 0; j < (y_extent); j++) {
    for (int i = 0; i < (x_extent); i++) {
      energy0[i + j * cell_row_pitch] = background_energy;
      density0[i + j * cell_row_pitch] = background_density;
      velx0[i + j * node_row_pitch] = background_velx;
      vely0[i + j * node_row_pitch] = background_vely;
    }
  }

  for (int region_id = 1; region_id < globals.settings.region_count; ++region_id) {

    double *xcentre = field.xcentre.data;
    double *ycentre = field.ycentre.data;

    double *xnode = field.xnode.data;
    double *ynode = field.ynode.data;

    const double *region_density = region_density_buf.data;
    const double *region_energy = region_energy_buf.data;
    const double *region_velx = region_velx_buf.data;
    const double *region_vely = region_vely_buf.data;
    const double *region_xlo = region_xlo_buf.data;
    const double *region_xhi = region_xhi_buf.data;
    const double *region_ylo = region_ylo_buf.data;
    const double *region_yhi = region_yhi_buf.data;
    const double *region_radius = region_radius_buf.data;
    const int *region_shape_code = region_shape_buf.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(globals.context.use_target)                                 \
    copyin(region_density[ : region_density_buf.N()]) copyin(region_energy[ : region_energy_buf.N()])                                \
    copyin(region_velx[ : region_velx_buf.N()]) copyin(region_vely[ : region_vely_buf.N()])                                          \
    copyin(region_xlo[ : region_xlo_buf.N()]) copyin(region_xhi[ : region_xhi_buf.N()])                                          \
    copyin(region_ylo[ : region_ylo_buf.N()]) copyin(region_yhi[ : region_yhi_buf.N()])                                          \
    copyin(region_radius[ : region_radius_buf.N()]) copyin(region_shape_code[ : region_shape_buf.N()])                              \
    present(energy0[ : field.energy0.N()], density0[ : field.density0.N()], velx0[ : field.velx0.N()], vely0[: field.vely0.N()],       \
            xcentre[ : field.xcentre.N()], ycentre[ : field.ycentre.N()], xnode[ : field.xnode.N()], ynode[ : field.ynode.N()])
    for (int j = 0; j < (y_extent); j++) {
      for (int i = 0; i < (x_extent); i++) {
        double centre_x = region_xlo[region_id];
        double centre_y = region_ylo[region_id];
        if (region_shape_code[region_id] == glb_rectangle) {
          if (xnode[i + 1] >= region_xlo[region_id] && xnode[i] < region_xhi[region_id]) {
            if (ynode[j + 1] >= region_ylo[region_id] && ynode[j] < region_yhi[region_id]) {
              energy0[i + j * cell_row_pitch] = region_energy[region_id];
              density0[i + j * cell_row_pitch] = region_density[region_id];
              for (int kt = j; kt <= j + 1; ++kt) {
                for (int jt = i; jt <= i + 1; ++jt) {
                  velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                  vely0[jt + kt * node_row_pitch] = region_vely[region_id];
                }
              }
            }
          }
        } else if (region_shape_code[region_id] == glb_circle) {
          double circle_radius = sqrt((xcentre[i] - centre_x) * (xcentre[i] - centre_x) + (ycentre[j] - centre_y) * (ycentre[j] - centre_y));
          if (circle_radius <= region_radius[region_id]) {
            energy0[i + j * cell_row_pitch] = region_energy[region_id];
            density0[i + j * cell_row_pitch] = region_density[region_id];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                vely0[jt + kt * node_row_pitch] = region_vely[region_id];
              }
            }
          }
        } else if (region_shape_code[region_id] == glb_single_point) {
          if (xnode[i] == centre_x && ynode[j] == centre_y) {
            energy0[i + j * cell_row_pitch] = region_energy[region_id];
            density0[i + j * cell_row_pitch] = region_density[region_id];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                vely0[jt + kt * node_row_pitch] = region_vely[region_id];
              }
            }
          }
        }
      }
    }
  }
}
