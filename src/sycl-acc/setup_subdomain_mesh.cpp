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

  const size_t x_extent = (x_hi + 3) - (x_lo - 2) + 1;
  const size_t y_extent = (y_hi + 3) - (y_lo - 2) + 1;

  // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.
  field_set &field = globals.subdomain.tile_list[tile_id].field;

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    auto xnode = field.xnode.access<W>(h);
    auto dxnode = field.dxnode.access<W>(h);
    hydro2d::par_ranged<class CONCAT_LINE(startup)>(h, {0u, x_extent}, [=](id<1> j) {
      xnode[j] = x_lower + dx * (static_cast<int>(j[0]) - 1 - x_lo);
      dxnode[j] = dx;
    });
  });

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    auto ynode = field.ynode.access<W>(h);
    auto dynode = field.dynode.access<W>(h);
    hydro2d::par_ranged<class CONCAT_LINE(startup)>(h, {0u, y_extent}, [=](id<1> k) {
      ynode[k] = y_lower + dy * (static_cast<int>(k[0]) - 1 - y_lo);
      dynode[k] = dy;
    });
  });

  const size_t x_extent_full = (x_hi + 2) - (x_lo - 2) + 1;
  const size_t y_extent_full = (y_hi + 2) - (y_lo - 2) + 1;

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    auto xcentre = field.xcentre.access<W>(h);
    auto dxcell = field.dxcell.access<W>(h);
    auto xnode = field.xnode.access<R>(h);
    hydro2d::par_ranged<class CONCAT_LINE(startup)>(h, {0u, x_extent_full}, [=](id<1> j) {
      xcentre[j] = 0.5 * (xnode[j] + xnode[j[0] + 1]);
      dxcell[j] = dx;
    });
  });

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    auto ycentre = field.ycentre.access<W>(h);
    auto dycell = field.dycell.access<W>(h);
    auto ynode = field.ynode.access<R>(h);

    hydro2d::par_ranged<class CONCAT_LINE(startup)>(h, {0u, y_extent_full}, [=](id<1> k) {
      ycentre[k] = 0.5 * (ynode[k] + ynode[k[0] + 1]);
      dycell[k] = dy;
    });
  });

  hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
    auto cell_volume = field.cell_volume.access<W>(h);
    auto xface_area = field.xface_area.access<W>(h);
    auto yface_area = field.yface_area.access<W>(h);
    auto dxcell = field.dxcell.access<R>(h);
    auto dycell = field.dycell.access<R>(h);
    hydro2d::par_ranged<class CONCAT_LINE(startup)>(h, {0u, 0u, x_extent_full, y_extent_full}, [=](id<2> idx) {
      cell_volume[idx] = dx * dy;
      xface_area[idx] = dycell[idx[1]];
      yface_area[idx] = dxcell[idx[0]];
    });
  });
}
