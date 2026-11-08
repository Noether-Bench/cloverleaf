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

void setup_subdomain_mesh(const int tile_id, sim_state &globals) {

  double dx = (globals.settings.mesh_spec.x_upper - globals.settings.mesh_spec.x_lower) / (double)(globals.settings.mesh_spec.grid_nx);
  double dy = (globals.settings.mesh_spec.y_upper - globals.settings.mesh_spec.y_lower) / (double)(globals.settings.mesh_spec.grid_ny);

  double x_lower = globals.settings.mesh_spec.x_lower + dx * (double)(globals.subdomain.tile_list[tile_id].meta.tile_west_idx - 1);

  double y_lower = globals.settings.mesh_spec.y_lower + dy * (double)(globals.subdomain.tile_list[tile_id].meta.blk_bottom - 1);

  ////    CALL initialise_chunk_kernel(subdomain%tile_list(tile_id)%tile_xlo,    &
  //     subdomain%tile_list(tile_id)%blk_xmax,    &
  //     subdomain%tile_list(tile_id)%blk_ymin,    &
  //     subdomain%tile_list(tile_id)%blk_ymax,    &
  //     x_lower,y_lower,dx,dy,              &
  //     subdomain%tile_list(tile_id)%field%xnode,  &
  //     subdomain%tile_list(tile_id)%field%dxnode, &
  //     subdomain%tile_list(tile_id)%field%ynode,  &
  //     subdomain%tile_list(tile_id)%field%dynode, &
  //     subdomain%tile_list(tile_id)%field%xcentre,    &
  //     subdomain%tile_list(tile_id)%field%dxcell,   &
  //     subdomain%tile_list(tile_id)%field%ycentre,    &
  //     subdomain%tile_list(tile_id)%field%dycell,   &
  //     subdomain%tile_list(tile_id)%field%cell_volume,   &
  //     subdomain%tile_list(tile_id)%field%xface_area,    &
  //     subdomain%tile_list(tile_id)%field%yface_area     )

  const int x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  const int x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  const int y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  const int y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  size_t x_extent = (x_hi + 3) - (x_lo - 2) + 1;
  size_t y_extent = (y_hi + 3) - (y_lo - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.
  field_set &field = globals.subdomain.tile_list[tile_id].field;

  Kokkos::parallel_for(
      x_extent, KOKKOS_LAMBDA(const int j) {
        field.xnode.host_vals(j) = x_lower + dx * (double)(j - 1 - x_lo);
        field.dxnode.host_vals(j) = dx;
      });

  Kokkos::parallel_for(
      y_extent, KOKKOS_LAMBDA(const int k) {
        field.ynode.host_vals(k) = y_lower + dy * (double)(k - 1 - y_lo);
        field.dynode.host_vals(k) = dy;
      });

  x_extent = (x_hi + 2) - (x_lo - 2) + 1;
  y_extent = (y_hi + 2) - (y_lo - 2) + 1;

  Kokkos::parallel_for(
      x_extent, KOKKOS_LAMBDA(const int j) {
        field.xcentre.host_vals(j) = 0.5 * (field.xnode.host_vals(j) + field.xnode.host_vals(j + 1));
        field.dxcell.host_vals(j) = dx;
      });

  Kokkos::parallel_for(
      y_extent, KOKKOS_LAMBDA(const int k) {
        field.ycentre.host_vals(k) = 0.5 * (field.ynode.host_vals(k) + field.ynode.host_vals(k + 1));
        field.dycell.host_vals(k) = dy;
      });

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {x_extent, y_extent}), KOKKOS_LAMBDA(const int j, const int k) {
        field.cell_volume.host_vals(j, k) = dx * dy;
        field.xface_area.host_vals(j, k) = field.dycell.host_vals(k);
        field.yface_area.host_vals(j, k) = field.dxcell.host_vals(j);
      });
}
