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

// @brief Driver for subdomain initialisation.
// @author Wayne Gaudin
// @details Invokes the user specified subdomain initialisation kernel.
// @brief Fortran subdomain initialisation kernel.
// @author Wayne Gaudin
// @details Calculates mesh shape_selector for the mesh subdomain based on the mesh size.

#include "setup_subdomain_mesh.h"
#include "context.h"

void setup_subdomain_mesh(const int tile_id, sim_state &globals) {

  double dx = (globals.settings.mesh_spec.x_upper - globals.settings.mesh_spec.x_lower) / (double)(globals.settings.mesh_spec.grid_nx);
  double dy = (globals.settings.mesh_spec.y_upper - globals.settings.mesh_spec.y_lower) / (double)(globals.settings.mesh_spec.grid_ny);

  double x_lower = globals.settings.mesh_spec.x_lower + dx * (double)(globals.subdomain.tile_list[tile_id].meta.tile_west_idx - 1);

  double y_lower = globals.settings.mesh_spec.y_lower + dy * (double)(globals.subdomain.tile_list[tile_id].meta.blk_bottom - 1);

  const int x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  const int x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  const int y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  const int y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  const size_t x_extent = (x_hi + 3) - (x_lo - 2) + 1;
  const size_t y_extent = (y_hi + 3) - (y_lo - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.
  field_set &field = globals.subdomain.tile_list[tile_id].field;

  hydro2d::par_ranged1(Span1d{0u, x_extent}, [=] DEVICE_KERNEL(int j) {
    field.xnode[j] = x_lower + dx * (static_cast<int>(j) - 1 - x_lo);
    field.dxnode[j] = dx;
  });

  hydro2d::par_ranged1(Span1d{0u, y_extent}, [=] DEVICE_KERNEL(int k) {
    field.ynode[k] = y_lower + dy * (static_cast<int>(k) - 1 - y_lo);
    field.dynode[k] = dy;
  });

  const size_t x_extent_full = (x_hi + 2) - (x_lo - 2) + 1;
  const size_t y_extent_full = (y_hi + 2) - (y_lo - 2) + 1;

  hydro2d::par_ranged1(Span1d{0u, x_extent_full}, [=] DEVICE_KERNEL(int j) {
    field.xcentre[j] = 0.5 * (field.xnode[j] + field.xnode[j + 1]);
    field.dxcell[j] = dx;
  });

  hydro2d::par_ranged1(Span1d{0u, y_extent_full}, [=] DEVICE_KERNEL(int k) {
    field.ycentre[k] = 0.5 * (field.ynode[k] + field.ynode[k + 1]);
    field.dycell[k] = dy;
  });

  hydro2d::par_ranged2(Span2d{0u, 0u, x_extent_full, y_extent_full}, [=] DEVICE_KERNEL(const int i, const int j) {
    field.cell_volume(i, j) = dx * dy;
    field.xface_area(i, j) = field.dycell[j];
    field.yface_area(i, j) = field.dxcell[i];
  });
}
