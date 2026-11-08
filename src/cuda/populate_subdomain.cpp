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
#include "context.h"
#include <cmath>

void populate_subdomain(const int tile_id, sim_state &globals) {

  // Need to copy the host array of region_id input data into a target_option array
  std::vector<double> state_density_vec(globals.settings.region_count);
  std::vector<double> state_energy_vec(globals.settings.region_count);
  std::vector<double> state_xvel_vec(globals.settings.region_count);
  std::vector<double> state_yvel_vec(globals.settings.region_count);
  std::vector<double> state_xmin_vec(globals.settings.region_count);
  std::vector<double> state_xmax_vec(globals.settings.region_count);
  std::vector<double> state_ymin_vec(globals.settings.region_count);
  std::vector<double> state_ymax_vec(globals.settings.region_count);
  std::vector<double> state_radius_vec(globals.settings.region_count);
  std::vector<int> state_geometry_vec(globals.settings.region_count);

  // Copy the data to the new views
  for (int region_id = 0; region_id < globals.settings.region_count; ++region_id) {
    state_density_vec[region_id] = globals.settings.regions[region_id].initial_density;
    state_energy_vec[region_id] = globals.settings.regions[region_id].initial_energy;
    state_xvel_vec[region_id] = globals.settings.regions[region_id].velx;
    state_yvel_vec[region_id] = globals.settings.regions[region_id].vely;
    state_xmin_vec[region_id] = globals.settings.regions[region_id].x_lower;
    state_xmax_vec[region_id] = globals.settings.regions[region_id].x_upper;
    state_ymin_vec[region_id] = globals.settings.regions[region_id].y_lower;
    state_ymax_vec[region_id] = globals.settings.regions[region_id].y_upper;
    state_radius_vec[region_id] = globals.settings.regions[region_id].circle_radius;
    state_geometry_vec[region_id] = globals.settings.regions[region_id].shape_selector;
  }

  // Need to copy the host array of region_id input data into a target_option array
  hydro2d::DeviceArray1D<double> region_density(globals.context, globals.settings.region_count, state_density_vec.data());
  hydro2d::DeviceArray1D<double> region_energy(globals.context, globals.settings.region_count, state_energy_vec.data());
  hydro2d::DeviceArray1D<double> region_velx(globals.context, globals.settings.region_count, state_xvel_vec.data());
  hydro2d::DeviceArray1D<double> region_vely(globals.context, globals.settings.region_count, state_yvel_vec.data());
  hydro2d::DeviceArray1D<double> region_xlo(globals.context, globals.settings.region_count, state_xmin_vec.data());
  hydro2d::DeviceArray1D<double> region_xhi(globals.context, globals.settings.region_count, state_xmax_vec.data());
  hydro2d::DeviceArray1D<double> region_ylo(globals.context, globals.settings.region_count, state_ymin_vec.data());
  hydro2d::DeviceArray1D<double> region_yhi(globals.context, globals.settings.region_count, state_ymax_vec.data());
  hydro2d::DeviceArray1D<double> region_radius(globals.context, globals.settings.region_count, state_radius_vec.data());
  hydro2d::DeviceArray1D<int> region_shape_code(globals.context, globals.settings.region_count, state_geometry_vec.data());

  const int x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  const int x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  const int y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  const int y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  size_t x_extent = (x_hi + 2) - (x_lo - 2) + 1;
  size_t y_extent = (y_hi + 2) - (y_lo - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.

  hydro2d::Span2d xyrange_policy(0u, 0u, x_extent, y_extent);

  field_set &field = globals.subdomain.tile_list[tile_id].field;

  // State 1 is always the background region_id

  hydro2d::par_ranged2(xyrange_policy, [=] DEVICE_KERNEL(const size_t i, const size_t j) {
    field.energy0(i, j) = region_energy[0];
    field.density0(i, j) = region_density[0];
    field.velx0(i, j) = region_velx[0];
    field.vely0(i, j) = region_vely[0];
  });

  for (int region_id = 1; region_id < globals.settings.region_count; ++region_id) {

    hydro2d::par_ranged2(xyrange_policy, [=] DEVICE_KERNEL(const size_t x, const size_t y) {
      const int j = x;
      const int k = y;

      double centre_x = region_xlo[region_id];
      double centre_y = region_ylo[region_id];

      if (region_shape_code[region_id] == glb_rectangle) {
        if (field.xnode[j + 1] >= region_xlo[region_id] && field.xnode[j] < region_xhi[region_id]) {
          if (field.ynode[k + 1] >= region_ylo[region_id] && field.ynode[k] < region_yhi[region_id]) {
            field.energy0(x, y) = region_energy[region_id];
            field.density0(x, y) = region_density[region_id];
            for (int kt = k; kt <= k + 1; ++kt) {
              for (int jt = j; jt <= j + 1; ++jt) {
                field.velx0(jt, kt) = region_velx[region_id];
                field.vely0(jt, kt) = region_vely[region_id];
              }
            }
          }
        }
      } else if (region_shape_code[region_id] == glb_circle) {
        double circle_radius =
            std::sqrt((field.xcentre[j] - centre_x) * (field.xcentre[j] - centre_x) + (field.ycentre[k] - centre_y) * (field.ycentre[k] - centre_y));
        if (circle_radius <= region_radius[region_id]) {
          field.energy0(x, y) = region_energy[region_id];
          field.density0(x, y) = region_density[region_id];
          for (int kt = k; kt <= k + 1; ++kt) {
            for (int jt = j; jt <= j + 1; ++jt) {
              field.velx0(jt, kt) = region_velx[region_id];
              field.vely0(jt, kt) = region_vely[region_id];
            }
          }
        }
      } else if (region_shape_code[region_id] == glb_single_point) {
        if (field.xnode[j] == centre_x && field.ynode[k] == centre_y) {
          field.energy0(x, y) = region_energy[region_id];
          field.density0(x, y) = region_density[region_id];
          for (int kt = k; kt <= k + 1; ++kt) {
            for (int jt = j; jt <= j + 1; ++jt) {
              field.velx0(jt, kt) = region_velx[region_id];
              field.vely0(jt, kt) = region_vely[region_id];
            }
          }
        }
      }
    });
  }

  region_density.release();
  region_energy.release();
  region_velx.release();
  region_vely.release();
  region_xlo.release();
  region_xhi.release();
  region_ylo.release();
  region_yhi.release();
  region_radius.release();
  region_shape_code.release();
}
