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

#include "refresh_ghost_cells.h"
#include "messaging_kernel.h"
#include "wall_seconds.h"
#include "exchange_tile_ghosts.h"

//   @brief Fortran kernel to update the external halo cell_total in a subdomain.
//   @author Wayne Gaudin
//   @details Updates halo cell_total for the required exchange_mask at the required halo_width
//   for any halo cell_total copied lie on an external boundary. The origin_site and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.
void refresh_ghost_cells_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                        const std::array<int, 4> &tile_adjacent, field_set &field, const int exchange_mask[FIELD_COUNT], int halo_width) {

  const size_t cell_row_pitch = field.cell_row_pitch;
  const size_t node_row_pitch = field.node_row_pitch;
  const size_t xflux_row_pitch = field.xflux_row_pitch;
  const size_t yface_row_stride = field.yface_row_stride;

  //  Update gathered in external halo cell_total based on halo_width and exchange_mask requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  copied halo_width is either 1 or 2 so copied it is more efficient to always thread
  //  loop along the mesh edge.
  if (exchange_mask[fld_density0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *density0 = field.density0.data;

#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density0[ : field.density0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          density0[j + (1 - k) * cell_row_pitch] = density0[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *density0 = field.density0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density0[ : field.density0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          density0[j + (y_hi + 2 + k) * cell_row_pitch] = density0[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *density0 = field.density0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density0[ : field.density0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          density0[(1 - j) + (k)*cell_row_pitch] = density0[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *density0 = field.density0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density0[ : field.density0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          density0[(x_hi + 2 + j) + (k)*cell_row_pitch] = density0[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_density1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *density1 = field.density1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density1[ : field.density1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          density1[j + (1 - k) * cell_row_pitch] = density1[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *density1 = field.density1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density1[ : field.density1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          density1[j + (y_hi + 2 + k) * cell_row_pitch] = density1[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *density1 = field.density1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density1[ : field.density1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          density1[(1 - j) + (k)*cell_row_pitch] = density1[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *density1 = field.density1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(density1[ : field.density1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          density1[(x_hi + 2 + j) + (k)*cell_row_pitch] = density1[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_energy0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      //  DO j=x_lo-halo_width,x_hi+halo_width

      double *energy0 = field.energy0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(energy0[ : field.energy0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          energy0[j + (1 - k) * cell_row_pitch] = energy0[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *energy0 = field.energy0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(energy0[ : field.energy0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          energy0[j + (y_hi + 2 + k) * cell_row_pitch] = energy0[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *energy0 = field.energy0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(energy0[ : field.energy0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          energy0[(1 - j) + (k)*cell_row_pitch] = energy0[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *energy0 = field.energy0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(energy0[ : field.energy0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          energy0[(x_hi + 2 + j) + (k)*cell_row_pitch] = energy0[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_energy1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *spec_energy1 = field.spec_energy1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(spec_energy1[ : field.spec_energy1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          spec_energy1[j + (1 - k) * cell_row_pitch] = spec_energy1[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *spec_energy1 = field.spec_energy1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(spec_energy1[ : field.spec_energy1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          spec_energy1[j + (y_hi + 2 + k) * cell_row_pitch] = spec_energy1[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *spec_energy1 = field.spec_energy1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(spec_energy1[ : field.spec_energy1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          spec_energy1[(1 - j) + (k)*cell_row_pitch] = spec_energy1[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *spec_energy1 = field.spec_energy1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(spec_energy1[ : field.spec_energy1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          spec_energy1[(x_hi + 2 + j) + (k)*cell_row_pitch] = spec_energy1[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_pressure] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *cell_pressure = field.cell_pressure.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(cell_pressure[ : field.cell_pressure.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          cell_pressure[j + (1 - k) * cell_row_pitch] = cell_pressure[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *cell_pressure = field.cell_pressure.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(cell_pressure[ : field.cell_pressure.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          cell_pressure[j + (y_hi + 2 + k) * cell_row_pitch] = cell_pressure[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *cell_pressure = field.cell_pressure.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(cell_pressure[ : field.cell_pressure.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          cell_pressure[(1 - j) + (k)*cell_row_pitch] = cell_pressure[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *cell_pressure = field.cell_pressure.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(cell_pressure[ : field.cell_pressure.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          cell_pressure[(x_hi + 2 + j) + (k)*cell_row_pitch] = cell_pressure[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_viscosity] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *shock_damping = field.shock_damping.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(shock_damping[ : field.shock_damping.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          shock_damping[j + (1 - k) * cell_row_pitch] = shock_damping[j + (2 + k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *shock_damping = field.shock_damping.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(shock_damping[ : field.shock_damping.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          shock_damping[j + (y_hi + 2 + k) * cell_row_pitch] = shock_damping[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *shock_damping = field.shock_damping.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(shock_damping[ : field.shock_damping.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          shock_damping[(1 - j) + (k)*cell_row_pitch] = shock_damping[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *shock_damping = field.shock_damping.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(shock_damping[ : field.shock_damping.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          shock_damping[(x_hi + 2 + j) + (k)*cell_row_pitch] = shock_damping[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_sound_speed] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *wave_speed = field.wave_speed.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(wave_speed[ : field.wave_speed.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          wave_speed[j + (1 - k) * cell_row_pitch] = wave_speed[j + (+k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *wave_speed = field.wave_speed.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(wave_speed[ : field.wave_speed.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          wave_speed[j + (y_hi + 2 + k) * cell_row_pitch] = wave_speed[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width

      double *wave_speed = field.wave_speed.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(wave_speed[ : field.wave_speed.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          wave_speed[(1 - j) + (k)*cell_row_pitch] = wave_speed[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width

      double *wave_speed = field.wave_speed.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(wave_speed[ : field.wave_speed.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          wave_speed[(x_hi + 2 + j) + (k)*cell_row_pitch] = wave_speed[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_velx0] == 1) {

    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *velx0 = field.velx0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx0[ : field.velx0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx0[j + (1 - k) * node_row_pitch] = velx0[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *velx0 = field.velx0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx0[ : field.velx0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx0[j + (y_hi + 1 + 2 + k) * node_row_pitch] = velx0[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *velx0 = field.velx0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx0[ : field.velx0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx0[(1 - j) + (k)*node_row_pitch] = -velx0[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *velx0 = field.velx0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx0[ : field.velx0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx0[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = -velx0[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_velx1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *velx1 = field.velx1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx1[ : field.velx1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx1[j + (1 - k) * node_row_pitch] = velx1[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *velx1 = field.velx1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx1[ : field.velx1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx1[j + (y_hi + 1 + 2 + k) * node_row_pitch] = velx1[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *velx1 = field.velx1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx1[ : field.velx1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx1[(1 - j) + (k)*node_row_pitch] = -velx1[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *velx1 = field.velx1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(velx1[ : field.velx1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx1[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = -velx1[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_vely0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *vely0 = field.vely0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely0[ : field.vely0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely0[j + (1 - k) * node_row_pitch] = -vely0[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *vely0 = field.vely0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely0[ : field.vely0.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely0[j + (y_hi + 1 + 2 + k) * node_row_pitch] = -vely0[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *vely0 = field.vely0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely0[ : field.vely0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely0[(1 - j) + (k)*node_row_pitch] = vely0[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *vely0 = field.vely0.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely0[ : field.vely0.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely0[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = vely0[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_vely1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *vely1 = field.vely1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely1[ : field.vely1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely1[j + (1 - k) * node_row_pitch] = -vely1[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *vely1 = field.vely1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely1[ : field.vely1.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely1[j + (y_hi + 1 + 2 + k) * node_row_pitch] = -vely1[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *vely1 = field.vely1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely1[ : field.vely1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely1[(1 - j) + (k)*node_row_pitch] = vely1[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *vely1 = field.vely1.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(vely1[ : field.vely1.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely1[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = vely1[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *volume_flux_x = field.volume_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_x[ : field.volume_flux_x.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_x[j + (1 - k) * xflux_row_pitch] = volume_flux_x[j + (1 + 2 + k) * xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *volume_flux_x = field.volume_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_x[ : field.volume_flux_x.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_x[j + (y_hi + 2 + k) * xflux_row_pitch] = volume_flux_x[j + (y_hi - k) * xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *volume_flux_x = field.volume_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_x[ : field.volume_flux_x.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_x[(1 - j) + (k)*xflux_row_pitch] = -volume_flux_x[(1 + 2 + j) + (k)*xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *volume_flux_x = field.volume_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_x[ : field.volume_flux_x.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_x[(x_hi + j + 1 + 2) + (k)*xflux_row_pitch] = -volume_flux_x[(x_hi + 1 - j) + (k)*xflux_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *face_mass_flux_x = field.face_mass_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_x[ : field.face_mass_flux_x.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_x[j + (1 - k) * xflux_row_pitch] = face_mass_flux_x[j + (1 + 2 + k) * xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      double *face_mass_flux_x = field.face_mass_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_x[ : field.volume_flux_x.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_x[j + (y_hi + 2 + k) * xflux_row_pitch] = face_mass_flux_x[j + (y_hi - k) * xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *face_mass_flux_x = field.face_mass_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_x[ : field.volume_flux_x.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_x[(1 - j) + (k)*xflux_row_pitch] = -face_mass_flux_x[(1 + 2 + j) + (k)*xflux_row_pitch];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      double *face_mass_flux_x = field.face_mass_flux_x.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_x[ : field.volume_flux_x.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_x[(x_hi + j + 1 + 2) + (k)*xflux_row_pitch] = -face_mass_flux_x[(x_hi + 1 - j) + (k)*xflux_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *volume_flux_y = field.volume_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_y[ : field.volume_flux_y.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_y[j + (1 - k) * yface_row_stride] = -volume_flux_y[j + (1 + 2 + k) * yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *volume_flux_y = field.volume_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_y[ : field.volume_flux_y.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_y[j + (y_hi + k + 1 + 2) * yface_row_stride] = -volume_flux_y[j + (y_hi + 1 - k) * yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *volume_flux_y = field.volume_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_y[ : field.volume_flux_y.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_y[(1 - j) + (k)*yface_row_stride] = volume_flux_y[(1 + 2 + j) + (k)*yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *volume_flux_y = field.volume_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(volume_flux_y[ : field.volume_flux_y.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_y[(x_hi + 2 + j) + (k)*yface_row_stride] = volume_flux_y[(x_hi - j) + (k)*yface_row_stride];
        }
      }
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *face_mass_flux_y = field.face_mass_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_y[ : field.face_mass_flux_y.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_y[j + (1 - k) * yface_row_stride] = -face_mass_flux_y[j + (1 + 2 + k) * yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      double *face_mass_flux_y = field.face_mass_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_y[ : field.face_mass_flux_y.N()])
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_y[j + (y_hi + k + 1 + 2) * yface_row_stride] = -face_mass_flux_y[j + (y_hi + 1 - k) * yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *face_mass_flux_y = field.face_mass_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_y[ : field.face_mass_flux_y.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_y[(1 - j) + (k)*yface_row_stride] = face_mass_flux_y[(1 + 2 + j) + (k)*yface_row_stride];
        }
      }
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      double *face_mass_flux_y = field.face_mass_flux_y.data;
#pragma acc parallel loop gang worker vector clover_use_target(use_target) \
  present(face_mass_flux_y[ : field.face_mass_flux_y.N()])
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_y[(x_hi + 2 + j) + (k)*yface_row_stride] = face_mass_flux_y[(x_hi - j) + (k)*yface_row_stride];
        }
      }
    }
  }
}

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cell_total for
//  the exchange_mask specified.
void refresh_ghost_cells(sim_state &globals, int exchange_mask[FIELD_COUNT], int halo_width) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();
  exchange_tile_ghosts(globals, exchange_mask, halo_width);
  if (globals.timing_enabled) {
    globals.kernel_timings.tile_halo_time += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  }

  hydro2d_halo_swap(globals, exchange_mask, halo_width);

  if (globals.timing_enabled) {
    globals.kernel_timings.rank_halo_time += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  }

  if ((globals.subdomain.subdomain_adjacent[subdomain_xlo] == mesh_edge) || (globals.subdomain.subdomain_adjacent[subdomain_xhi] == mesh_edge) ||
      (globals.subdomain.subdomain_adjacent[subdomain_ylo] == mesh_edge) || (globals.subdomain.subdomain_adjacent[subdomain_yhi] == mesh_edge)) {

#if HOST_DEVICE_SYNC
    globals.pushToDevice();
#endif

    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      tile_block &t = globals.subdomain.tile_list[tile_id];
      refresh_ghost_cells_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                         globals.subdomain.subdomain_adjacent, t.meta.tile_adjacent, t.field, exchange_mask, halo_width);
    }

#if HOST_DEVICE_SYNC
    globals.pullFromDevice();
#endif
  }

  if (globals.timing_enabled) globals.kernel_timings.local_halo_time += wall_seconds() - phase_start;
}
