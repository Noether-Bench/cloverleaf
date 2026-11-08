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

// @brief  Allocates the data for each mesh subdomain
// @author Wayne Gaudin
// @details The data exchange_mask for the mesh subdomain are allocated based on the mesh
// size.

#include "allocate_fields.h"

// Allocate Kokkos Views for the data arrays
void allocate_fields(sim_state &globals) {

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    const size_t x_extent = (globals.subdomain.tile_list[tile_id].meta.blk_xmax + 2) - (globals.subdomain.tile_list[tile_id].meta.tile_xlo - 2) + 1;
    const size_t y_extent = (globals.subdomain.tile_list[tile_id].meta.blk_ymax + 2) - (globals.subdomain.tile_list[tile_id].meta.blk_ymin - 2) + 1;

    // (tile_xlo-2:blk_xmax+2, blk_ymin-2:blk_ymax+2)
    new (&globals.subdomain.tile_list[tile_id].field.density0)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("density0"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.density1)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("density1"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.energy0)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("energy0"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.spec_energy1)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("spec_energy1"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.cell_pressure)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("cell_pressure"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.shock_damping)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("shock_damping"), x_extent, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.wave_speed)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("wave_speed"), x_extent, y_extent);

    // (tile_xlo-2:blk_xmax+3, blk_ymin-2:blk_ymax+3)
    new (&globals.subdomain.tile_list[tile_id].field.velx0)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("velx0"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.velx1)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("velx1"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.vely0)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("vely0"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.vely1)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("vely1"), x_extent + 1, y_extent + 1);

    // (tile_xlo-2:blk_xmax+3, blk_ymin-2:blk_ymax+2)
    new (&globals.subdomain.tile_list[tile_id].field.volume_flux_x)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("volume_flux_x"), x_extent + 1, y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.face_mass_flux_x)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("face_mass_flux_x"), x_extent + 1, y_extent);
    // (tile_xlo-2:blk_xmax+2, blk_ymin-2:blk_ymax+3)
    new (&globals.subdomain.tile_list[tile_id].field.volume_flux_y)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("volume_flux_y"), x_extent, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.face_mass_flux_y)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("face_mass_flux_y"), x_extent, y_extent + 1);

    // (tile_xlo-2:blk_xmax+3, blk_ymin-2:blk_ymax+3)
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf1)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf1"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf2)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf2"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf3)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf3"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf4)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf4"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf5)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf5"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf6)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf6"), x_extent + 1, y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.scratch_buf7)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("scratch_buf7"), x_extent + 1, y_extent + 1);

    // (tile_xlo-2:blk_xmax+2)
    new (&globals.subdomain.tile_list[tile_id].field.xcentre) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("xcentre"), x_extent);
    new (&globals.subdomain.tile_list[tile_id].field.dxcell) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("dxcell"), x_extent);
    // (blk_ymin-2:blk_ymax+2)
    new (&globals.subdomain.tile_list[tile_id].field.ycentre) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("ycentre"), y_extent);
    new (&globals.subdomain.tile_list[tile_id].field.dycell) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("dycell"), y_extent);
    // (tile_xlo-2:blk_xmax+3)
    new (&globals.subdomain.tile_list[tile_id].field.xnode) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("xnode"), x_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.dxnode) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("dxnode"), x_extent + 1);
    // (blk_ymin-2:blk_ymax+3)
    new (&globals.subdomain.tile_list[tile_id].field.ynode) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("ynode"), y_extent + 1);
    new (&globals.subdomain.tile_list[tile_id].field.dynode) Kokkos::View<double *>(Kokkos::ViewAllocateWithoutInitializing("dynode"), y_extent + 1);

    // (tile_xlo-2:blk_xmax+2, blk_ymin-2:blk_ymax+2)
    new (&globals.subdomain.tile_list[tile_id].field.cell_volume)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("cell_volume"), x_extent, y_extent);
    // (tile_xlo-2:blk_xmax+3, blk_ymin-2:blk_ymax+2)
    new (&globals.subdomain.tile_list[tile_id].field.xface_area)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("xface_area"), x_extent + 1, y_extent);
    // (tile_xlo-2:blk_xmax+2, blk_ymin-2:blk_ymax+3)
    new (&globals.subdomain.tile_list[tile_id].field.yface_area)
        Kokkos::View<double **>(Kokkos::ViewAllocateWithoutInitializing("yface_area"), x_extent, y_extent + 1);

    // Zeroing isn't strictly neccessary but it ensures physical pages
    // are allocated. This prevents first touch overheads in the main code
    // cycle which can skew timings in the first cycle_index

    // Take a reference to the lowest structure, as Kokkos target_option cannot necessarily chase through the structure.
    field_set &field = globals.subdomain.tile_list[tile_id].field;

    // Nested loop over (blk_ymin-2:blk_ymax+3) and (tile_xlo-2:blk_xmax+3) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_1({0, 0}, {x_extent + 1, y_extent + 1});

    Kokkos::parallel_for(
        "build_field_zero_1", loop_bounds_1, KOKKOS_LAMBDA(const int j, const int k) {
          field.scratch_buf1.host_vals(j, k) = 0.0;
          field.scratch_buf2.host_vals(j, k) = 0.0;
          field.scratch_buf3.host_vals(j, k) = 0.0;
          field.scratch_buf4.host_vals(j, k) = 0.0;
          field.scratch_buf5.host_vals(j, k) = 0.0;
          field.scratch_buf6.host_vals(j, k) = 0.0;
          field.scratch_buf7.host_vals(j, k) = 0.0;

          field.velx0.host_vals(j, k) = 0.0;
          field.velx1.host_vals(j, k) = 0.0;
          field.vely0.host_vals(j, k) = 0.0;
          field.vely1.host_vals(j, k) = 0.0;
        });

    // Nested loop over (blk_ymin-2:blk_ymax+2) and (tile_xlo-2:blk_xmax+2) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_2({0, 0}, {x_extent, y_extent});

    Kokkos::parallel_for(
        "build_field_zero_2", loop_bounds_2, KOKKOS_LAMBDA(const int j, const int k) {
          field.density0.host_vals(j, k) = 0.0;
          field.density1.host_vals(j, k) = 0.0;
          field.energy0.host_vals(j, k) = 0.0;
          field.spec_energy1.host_vals(j, k) = 0.0;
          field.cell_pressure.host_vals(j, k) = 0.0;
          field.shock_damping.host_vals(j, k) = 0.0;
          field.wave_speed.host_vals(j, k) = 0.0;
          field.cell_volume.host_vals(j, k) = 0.0;
        });

    // Nested loop over (blk_ymin-2:blk_ymax+2) and (tile_xlo-2:blk_xmax+3) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_3({0, 0}, {x_extent + 1, y_extent});

    Kokkos::parallel_for(
        "build_field_zero_3", loop_bounds_3, KOKKOS_LAMBDA(const int j, const int k) {
          field.volume_flux_x.host_vals(j, k) = 0.0;
          field.face_mass_flux_x.host_vals(j, k) = 0.0;
          field.xface_area.host_vals(j, k) = 0.0;
        });

    // Nested loop over (blk_ymin-2:blk_ymax+3) and (tile_xlo-2:blk_xmax+2) inclusive
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> loop_bounds_4({0, 0}, {x_extent, y_extent + 1});

    Kokkos::parallel_for(
        "build_field_zero_4", loop_bounds_4, KOKKOS_LAMBDA(const int j, const int k) {
          field.volume_flux_y.host_vals(j, k) = 0.0;
          field.face_mass_flux_y.host_vals(j, k) = 0.0;
          field.yface_area.host_vals(j, k) = 0.0;
        });

    // (tile_xlo-2:blk_xmax+2) inclusive
    Kokkos::parallel_for(
        "build_field_zero_5", x_extent, KOKKOS_LAMBDA(const int j) {
          field.xcentre.host_vals(j) = 0.0;
          field.dxcell.host_vals(j) = 0.0;
        });

    // (blk_ymin-2:blk_ymax+2) inclusive
    Kokkos::parallel_for(
        "build_field_zero_6", y_extent, KOKKOS_LAMBDA(const int k) {
          field.ycentre.host_vals(k) = 0.0;
          field.dycell.host_vals(k) = 0.0;
        });

    // (tile_xlo-2:blk_xmax+3) inclusive
    Kokkos::parallel_for(
        "build_field_zero_6", x_extent + 1, KOKKOS_LAMBDA(const int j) {
          field.xnode.host_vals(j) = 0.0;
          field.dxnode.host_vals(j) = 0.0;
        });

    // (blk_ymin-2:blk_ymax+3) inclusive
    Kokkos::parallel_for(
        "build_field_zero_7", y_extent + 1, KOKKOS_LAMBDA(const int k) {
          field.ynode.host_vals(k) = 0.0;
          field.dynode.host_vals(k) = 0.0;
        });
  }
}
