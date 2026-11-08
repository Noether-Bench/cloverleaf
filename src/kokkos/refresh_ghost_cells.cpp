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
#include "messaging.h"
#include "messaging_kernel.h"
#include "wall_seconds.h"
#include "exchange_tile_ghosts.h"

//   @brief Fortran kernel to update the external halo cell_total in a subdomain.
//   @author Wayne Gaudin
//   @details Updates halo cell_total for the required exchange_mask at the required halo_width
//   for any halo cell_total copied lie on an external boundary. The origin_site and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.
void refresh_ghost_cells_kernel(int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                        const std::array<int, 4> &tile_adjacent, Kokkos::View<double **> &density0, Kokkos::View<double **> &energy0,
                        Kokkos::View<double **> &cell_pressure, Kokkos::View<double **> &shock_damping, Kokkos::View<double **> &wave_speed,
                        Kokkos::View<double **> &density1, Kokkos::View<double **> &spec_energy1, Kokkos::View<double **> &velx0,
                        Kokkos::View<double **> &vely0, Kokkos::View<double **> &velx1, Kokkos::View<double **> &vely1,
                        Kokkos::View<double **> &volume_flux_x, Kokkos::View<double **> &volume_flux_y, Kokkos::View<double **> &face_mass_flux_x,
                        Kokkos::View<double **> &face_mass_flux_y, int exchange_mask[FIELD_COUNT], int halo_width) {

  //  Update gathered in external halo cell_total based on halo_width and exchange_mask requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  copied halo_width is either 1 or 2 so copied it is more efficient to always thread
  //  loop along the mesh edge.
  if (exchange_mask[fld_density0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              density0(j, 1 - k) = density0(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              density0(j, y_hi + 2 + k) = density0(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              density0(1 - j, k) = density0(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              density0(x_hi + 2 + j, k) = density0(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_density1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              density1(j, 1 - k) = density1(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              density1(j, y_hi + 2 + k) = density1(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              density1(1 - j, k) = density1(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              density1(x_hi + 2 + j, k) = density1(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_energy0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      //  DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              energy0(j, 1 - k) = energy0(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              energy0(j, y_hi + 2 + k) = energy0(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              energy0(1 - j, k) = energy0(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              energy0(x_hi + 2 + j, k) = energy0(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_energy1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              spec_energy1(j, 1 - k) = spec_energy1(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              spec_energy1(j, y_hi + 2 + k) = spec_energy1(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              spec_energy1(1 - j, k) = spec_energy1(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              spec_energy1(x_hi + 2 + j, k) = spec_energy1(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_pressure] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              cell_pressure(j, 1 - k) = cell_pressure(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              cell_pressure(j, y_hi + 2 + k) = cell_pressure(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              cell_pressure(1 - j, k) = cell_pressure(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              cell_pressure(x_hi + 2 + j, k) = cell_pressure(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_viscosity] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              shock_damping(j, 1 - k) = shock_damping(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              shock_damping(j, y_hi + 2 + k) = shock_damping(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              shock_damping(1 - j, k) = shock_damping(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              shock_damping(x_hi + 2 + j, k) = shock_damping(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_sound_speed] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              wave_speed(j, 1 - k) = wave_speed(j, 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              wave_speed(j, y_hi + 2 + k) = wave_speed(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              wave_speed(1 - j, k) = wave_speed(2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              wave_speed(x_hi + 2 + j, k) = wave_speed(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_velx0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              velx0(j, 1 - k) = velx0(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              velx0(j, y_hi + 1 + 2 + k) = velx0(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              velx0(1 - j, k) = -velx0(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              velx0(x_hi + 2 + 1 + j, k) = -velx0(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_velx1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              velx1(j, 1 - k) = velx1(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              velx1(j, y_hi + 1 + 2 + k) = velx1(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              velx1(1 - j, k) = -velx1(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              velx1(x_hi + 2 + 1 + j, k) = -velx1(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_vely0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              vely0(j, 1 - k) = -vely0(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              vely0(j, y_hi + 1 + 2 + k) = -vely0(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              vely0(1 - j, k) = vely0(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              vely0(x_hi + 2 + 1 + j, k) = vely0(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_vely1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              vely1(j, 1 - k) = -vely1(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              vely1(j, y_hi + 1 + 2 + k) = -vely1(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              vely1(1 - j, k) = vely1(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              vely1(x_hi + 2 + 1 + j, k) = vely1(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              volume_flux_x(j, 1 - k) = volume_flux_x(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              volume_flux_x(j, y_hi + 2 + k) = volume_flux_x(j, y_hi - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              volume_flux_x(1 - j, k) = -volume_flux_x(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              volume_flux_x(x_hi + j + 1 + 2, k) = -volume_flux_x(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              face_mass_flux_x(j, 1 - k) = face_mass_flux_x(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              face_mass_flux_x(j, y_hi + 2 + k) = face_mass_flux_x(j, y_hi - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              face_mass_flux_x(1 - j, k) = -face_mass_flux_x(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              face_mass_flux_x(x_hi + j + 1 + 2, k) = -face_mass_flux_x(x_hi + 1 - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              volume_flux_y(j, 1 - k) = -volume_flux_y(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              volume_flux_y(j, y_hi + k + 1 + 2) = -volume_flux_y(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              volume_flux_y(1 - j, k) = volume_flux_y(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              volume_flux_y(x_hi + 2 + j, k) = volume_flux_y(x_hi - j, k);
            }
          });
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              face_mass_flux_y(j, 1 - k) = -face_mass_flux_y(j, 1 + 2 + k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2), KOKKOS_LAMBDA(const int j) {
            for (int k = 0; k < halo_width; ++k) {
              face_mass_flux_y(j, y_hi + k + 1 + 2) = -face_mass_flux_y(j, y_hi + 1 - k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              face_mass_flux_y(1 - j, k) = face_mass_flux_y(1 + 2 + j, k);
            }
          });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width
      Kokkos::parallel_for(
          Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
            for (int j = 0; j < halo_width; ++j) {
              face_mass_flux_y(x_hi + 2 + j, k) = face_mass_flux_y(x_hi - j, k);
            }
          });
    }
  }
}

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cell_total for
//  the exchange_mask specified.
void refresh_ghost_cells(sim_state &globals, int exchange_mask[FIELD_COUNT], const int halo_width) {

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

    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      tile_block &t = globals.subdomain.tile_list[tile_id];
      refresh_ghost_cells_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.subdomain.subdomain_adjacent, t.meta.tile_adjacent,
                         t.field.density0.host_vals,    //
                         t.field.energy0.host_vals,     //
                         t.field.cell_pressure.host_vals,    //
                         t.field.shock_damping.host_vals,   //
                         t.field.wave_speed.host_vals,  //
                         t.field.density1.host_vals,    //
                         t.field.spec_energy1.host_vals,     //
                         t.field.velx0.host_vals,       //
                         t.field.vely0.host_vals,       //
                         t.field.velx1.host_vals,       //
                         t.field.vely1.host_vals,       //
                         t.field.volume_flux_x.host_vals,  //
                         t.field.volume_flux_y.host_vals,  //
                         t.field.face_mass_flux_x.host_vals, //
                         t.field.face_mass_flux_y.host_vals, //
                         exchange_mask, halo_width);
    }
  }

  if (globals.timing_enabled) globals.kernel_timings.local_halo_time += wall_seconds() - phase_start;
}
