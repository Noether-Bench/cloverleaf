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

#include "tile_ghost_copy_kernel.h"

//   @brief Fortran kernel to update the external halo cell_total in a subdomain.
//   @author Wayne Gaudin
//   @details Updates halo cell_total for the required exchange_mask at the required halo_width
//   for any halo cell_total copied lie on an external boundary. The origin_site and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.

void copy_tile_ghosts_west_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0_,
    hydro2d::DeviceArray2D<double> &energy0_, hydro2d::DeviceArray2D<double> &pressure_, hydro2d::DeviceArray2D<double> &viscosity_,
    hydro2d::DeviceArray2D<double> &soundspeed_, hydro2d::DeviceArray2D<double> &density1_, hydro2d::DeviceArray2D<double> &energy1_,
    hydro2d::DeviceArray2D<double> &xvel0_, hydro2d::DeviceArray2D<double> &yvel0_, hydro2d::DeviceArray2D<double> &xvel1_, hydro2d::DeviceArray2D<double> &yvel1_,
    hydro2d::DeviceArray2D<double> &vol_flux_x_, hydro2d::DeviceArray2D<double> &vol_flux_y_, hydro2d::DeviceArray2D<double> &mass_flux_x_,
    hydro2d::DeviceArray2D<double> &mass_flux_y_, int west_xlo, int west_xhi, int west_ylo, int west_yhi,
    hydro2d::DeviceArray2D<double> &left_density0_, hydro2d::DeviceArray2D<double> &left_energy0_, hydro2d::DeviceArray2D<double> &left_pressure_,
    hydro2d::DeviceArray2D<double> &left_viscosity_, hydro2d::DeviceArray2D<double> &left_soundspeed_, hydro2d::DeviceArray2D<double> &left_density1_,
    hydro2d::DeviceArray2D<double> &left_energy1_, hydro2d::DeviceArray2D<double> &left_xvel0_, hydro2d::DeviceArray2D<double> &left_yvel0_,
    hydro2d::DeviceArray2D<double> &left_xvel1_, hydro2d::DeviceArray2D<double> &left_yvel1_, hydro2d::DeviceArray2D<double> &left_vol_flux_x_,
    hydro2d::DeviceArray2D<double> &left_vol_flux_y_, hydro2d::DeviceArray2D<double> &left_mass_flux_x_, hydro2d::DeviceArray2D<double> &left_mass_flux_y_,
    const int exchange_mask[FIELD_COUNT], int halo_width) {

  auto density0 = density0_.host_vals;
  auto energy0 = energy0_.host_vals;
  auto cell_pressure = pressure_.host_vals;
  auto shock_damping = viscosity_.host_vals;
  auto wave_speed = soundspeed_.host_vals;
  auto density1 = density1_.host_vals;
  auto spec_energy1 = energy1_.host_vals;
  auto velx0 = xvel0_.host_vals;
  auto vely0 = yvel0_.host_vals;
  auto velx1 = xvel1_.host_vals;
  auto vely1 = yvel1_.host_vals;
  auto volume_flux_x = vol_flux_x_.host_vals;
  auto volume_flux_y = vol_flux_y_.host_vals;
  auto face_mass_flux_x = mass_flux_x_.host_vals;
  auto face_mass_flux_y = mass_flux_y_.host_vals;
  auto west_density0 = left_density0_.host_vals;
  auto west_energy0 = left_energy0_.host_vals;
  auto west_pressure = left_pressure_.host_vals;
  auto west_viscosity = left_viscosity_.host_vals;
  auto west_sound_speed = left_soundspeed_.host_vals;
  auto west_density1 = left_density1_.host_vals;
  auto west_energy1 = left_energy1_.host_vals;
  auto west_velx0 = left_xvel0_.host_vals;
  auto west_vely0 = left_yvel0_.host_vals;
  auto west_velx1 = left_xvel1_.host_vals;
  auto west_vely1 = left_yvel1_.host_vals;
  auto west_volume_flux_x = left_vol_flux_x_.host_vals;
  auto west_volume_flux_y = left_vol_flux_y_.host_vals;
  auto west_mass_flux_x = left_mass_flux_x_.host_vals;
  auto west_mass_flux_y = left_mass_flux_y_.host_vals;

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l density0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            density0(x_lo - j, k) = west_density0(west_xhi + 1 - j, k);
          }
        });
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l density1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            density1(x_lo - j, k) = west_density1(west_xhi + 1 - j, k);
          }
        });
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l energy0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            energy0(x_lo - j, k) = west_energy0(west_xhi + 1 - j, k);
          }
        });
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l spec_energy1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            spec_energy1(x_lo - j, k) = west_energy1(west_xhi + 1 - j, k);
          }
        });
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l cell_pressure", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            cell_pressure(x_lo - j, k) = west_pressure(west_xhi + 1 - j, k);
          }
        });
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l shock_damping", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            shock_damping(x_lo - j, k) = west_viscosity(west_xhi + 1 - j, k);
          }
        });
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l wave_speed", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            wave_speed(x_lo - j, k) = west_sound_speed(west_xhi + 1 - j, k);
          }
        });
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l velx0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            velx0(x_lo - j, k) = west_velx0(west_xhi + 1 - j, k);
          }
        });
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l velx1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            velx1(x_lo - j, k) = west_velx1(west_xhi + 1 - j, k);
          }
        });
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l vely0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            vely0(x_lo - j, k) = west_vely0(west_xhi + 1 - j, k);
          }
        });
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l vely1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            vely1(x_lo - j, k) = west_vely1(west_xhi + 1 - j, k);
          }
        });
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l volume_flux_x", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            volume_flux_x(x_lo - j, k) = west_volume_flux_x(west_xhi + 1 - j, k);
          }
        });
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l face_mass_flux_x", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            face_mass_flux_x(x_lo - j, k) = west_mass_flux_x(west_xhi + 1 - j, k);
          }
        });
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l volume_flux_y", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            volume_flux_y(x_lo - j, k) = west_volume_flux_y(west_xhi + 1 - j, k);
          }
        });
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_l face_mass_flux_y", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            face_mass_flux_y(x_lo - j, k) = west_mass_flux_y(west_xhi + 1 - j, k);
          }
        });
  }
}

void copy_tile_ghosts_east_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0_,
    hydro2d::DeviceArray2D<double> &energy0_, hydro2d::DeviceArray2D<double> &pressure_, hydro2d::DeviceArray2D<double> &viscosity_,
    hydro2d::DeviceArray2D<double> &soundspeed_, hydro2d::DeviceArray2D<double> &density1_, hydro2d::DeviceArray2D<double> &energy1_,
    hydro2d::DeviceArray2D<double> &xvel0_, hydro2d::DeviceArray2D<double> &yvel0_, hydro2d::DeviceArray2D<double> &xvel1_, hydro2d::DeviceArray2D<double> &yvel1_,
    hydro2d::DeviceArray2D<double> &vol_flux_x_, hydro2d::DeviceArray2D<double> &vol_flux_y_, hydro2d::DeviceArray2D<double> &mass_flux_x_,
    hydro2d::DeviceArray2D<double> &mass_flux_y_, int east_xlo, int east_xhi, int east_ylo, int east_yhi,
    hydro2d::DeviceArray2D<double> &right_density0_, hydro2d::DeviceArray2D<double> &right_energy0_, hydro2d::DeviceArray2D<double> &right_pressure_,
    hydro2d::DeviceArray2D<double> &right_viscosity_, hydro2d::DeviceArray2D<double> &right_soundspeed_, hydro2d::DeviceArray2D<double> &right_density1_,
    hydro2d::DeviceArray2D<double> &right_energy1_, hydro2d::DeviceArray2D<double> &right_xvel0_, hydro2d::DeviceArray2D<double> &right_yvel0_,
    hydro2d::DeviceArray2D<double> &right_xvel1_, hydro2d::DeviceArray2D<double> &right_yvel1_, hydro2d::DeviceArray2D<double> &right_vol_flux_x_,
    hydro2d::DeviceArray2D<double> &right_vol_flux_y_, hydro2d::DeviceArray2D<double> &right_mass_flux_x_, hydro2d::DeviceArray2D<double> &right_mass_flux_y_,
    const int exchange_mask[FIELD_COUNT], int halo_width) {

  auto density0 = density0_.host_vals;
  auto energy0 = energy0_.host_vals;
  auto cell_pressure = pressure_.host_vals;
  auto shock_damping = viscosity_.host_vals;
  auto wave_speed = soundspeed_.host_vals;
  auto density1 = density1_.host_vals;
  auto spec_energy1 = energy1_.host_vals;
  auto velx0 = xvel0_.host_vals;
  auto vely0 = yvel0_.host_vals;
  auto velx1 = xvel1_.host_vals;
  auto vely1 = yvel1_.host_vals;
  auto volume_flux_x = vol_flux_x_.host_vals;
  auto volume_flux_y = vol_flux_y_.host_vals;
  auto face_mass_flux_x = mass_flux_x_.host_vals;
  auto face_mass_flux_y = mass_flux_y_.host_vals;
  auto east_density0 = right_density0_.host_vals;
  auto east_energy0 = right_energy0_.host_vals;
  auto east_pressure = right_pressure_.host_vals;
  auto east_viscosity = right_viscosity_.host_vals;
  auto east_sound_speed = right_soundspeed_.host_vals;
  auto east_density1 = right_density1_.host_vals;
  auto east_energy1 = right_energy1_.host_vals;
  auto east_velx0 = right_xvel0_.host_vals;
  auto east_vely0 = right_yvel0_.host_vals;
  auto east_velx1 = right_xvel1_.host_vals;
  auto east_vely1 = right_yvel1_.host_vals;
  auto east_volume_flux_x = right_vol_flux_x_.host_vals;
  auto east_volume_flux_y = right_vol_flux_y_.host_vals;
  auto east_mass_flux_x = right_mass_flux_x_.host_vals;
  auto east_mass_flux_y = right_mass_flux_y_.host_vals;

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r density0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            density0(x_hi + 2 + j, k) = east_density0(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r density1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            density1(x_hi + 2 + j, k) = east_density1(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r energy0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            energy0(x_hi + 2 + j, k) = east_energy0(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r spec_energy1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            spec_energy1(x_hi + 2 + j, k) = east_energy1(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r cell_pressure", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            cell_pressure(x_hi + 2 + j, k) = east_pressure(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r shock_damping", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            shock_damping(x_hi + 2 + j, k) = east_viscosity(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r wave_speed", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            wave_speed(x_hi + 2 + j, k) = east_sound_speed(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r velx0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            velx0(x_hi + 1 + 2 + j, k) = east_velx0(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r velx1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            velx1(x_hi + 1 + 2 + j, k) = east_velx1(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r vely0", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            vely0(x_hi + 1 + 2 + j, k) = east_vely0(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r vely1", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            vely1(x_hi + 1 + 2 + j, k) = east_vely1(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r volume_flux_x", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            volume_flux_x(x_hi + 1 + 2 + j, k) = east_volume_flux_x(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r face_mass_flux_x", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            face_mass_flux_x(x_hi + 1 + 2 + j, k) = east_mass_flux_x(east_xlo + 1 - 1 + 2 + j, k);
          }
        });
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r volume_flux_y", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            volume_flux_y(x_hi + 2 + j, k) = east_volume_flux_y(east_xlo - 1 + 2 + j, k);
          }
        });
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    Kokkos::parallel_for(
        "update_tile_halo_r face_mass_flux_y", Kokkos::RangePolicy<>(y_lo - halo_width + 1, y_hi + 1 + halo_width + 2), KOKKOS_LAMBDA(const int k) {
          for (int j = 0; j < halo_width; ++j) {
            face_mass_flux_y(x_hi + 2 + j, k) = east_mass_flux_y(east_xlo - 1 + 2 + j, k);
          }
        });
  }
}

//  Top and first_row only do x_lower -> x_upper
//  This is because the corner ghosts will get communicated in the first_col last_col communication

void copy_tile_ghosts_north_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0_,
    hydro2d::DeviceArray2D<double> &energy0_, hydro2d::DeviceArray2D<double> &pressure_, hydro2d::DeviceArray2D<double> &viscosity_,
    hydro2d::DeviceArray2D<double> &soundspeed_, hydro2d::DeviceArray2D<double> &density1_, hydro2d::DeviceArray2D<double> &energy1_,
    hydro2d::DeviceArray2D<double> &xvel0_, hydro2d::DeviceArray2D<double> &yvel0_, hydro2d::DeviceArray2D<double> &xvel1_, hydro2d::DeviceArray2D<double> &yvel1_,
    hydro2d::DeviceArray2D<double> &vol_flux_x_, hydro2d::DeviceArray2D<double> &vol_flux_y_, hydro2d::DeviceArray2D<double> &mass_flux_x_,
    hydro2d::DeviceArray2D<double> &mass_flux_y_, int north_xlo, int north_xhi, int north_ylo, int north_yhi, hydro2d::DeviceArray2D<double> &top_density0_,
    hydro2d::DeviceArray2D<double> &top_energy0_, hydro2d::DeviceArray2D<double> &top_pressure_, hydro2d::DeviceArray2D<double> &top_viscosity_,
    hydro2d::DeviceArray2D<double> &top_soundspeed_, hydro2d::DeviceArray2D<double> &top_density1_, hydro2d::DeviceArray2D<double> &top_energy1_,
    hydro2d::DeviceArray2D<double> &top_xvel0_, hydro2d::DeviceArray2D<double> &top_yvel0_, hydro2d::DeviceArray2D<double> &top_xvel1_,
    hydro2d::DeviceArray2D<double> &top_yvel1_, hydro2d::DeviceArray2D<double> &top_vol_flux_x_, hydro2d::DeviceArray2D<double> &top_vol_flux_y_,
    hydro2d::DeviceArray2D<double> &top_mass_flux_x_, hydro2d::DeviceArray2D<double> &top_mass_flux_y_, const int exchange_mask[FIELD_COUNT], int halo_width) {

  auto density0 = density0_.host_vals;
  auto energy0 = energy0_.host_vals;
  auto cell_pressure = pressure_.host_vals;
  auto shock_damping = viscosity_.host_vals;
  auto wave_speed = soundspeed_.host_vals;
  auto density1 = density1_.host_vals;
  auto spec_energy1 = energy1_.host_vals;
  auto velx0 = xvel0_.host_vals;
  auto vely0 = yvel0_.host_vals;
  auto velx1 = xvel1_.host_vals;
  auto vely1 = yvel1_.host_vals;
  auto volume_flux_x = vol_flux_x_.host_vals;
  auto volume_flux_y = vol_flux_y_.host_vals;
  auto face_mass_flux_x = mass_flux_x_.host_vals;
  auto face_mass_flux_y = mass_flux_y_.host_vals;
  auto north_density0 = top_density0_.host_vals;
  auto north_energy0 = top_energy0_.host_vals;
  auto north_pressure = top_pressure_.host_vals;
  auto north_viscosity = top_viscosity_.host_vals;
  auto north_sound_speed = top_soundspeed_.host_vals;
  auto north_density1 = top_density1_.host_vals;
  auto north_energy1 = top_energy1_.host_vals;
  auto north_velx0 = top_xvel0_.host_vals;
  auto north_vely0 = top_yvel0_.host_vals;
  auto north_velx1 = top_xvel1_.host_vals;
  auto north_vely1 = top_yvel1_.host_vals;
  auto north_volume_flux_x = top_vol_flux_x_.host_vals;
  auto north_volume_flux_y = top_vol_flux_y_.host_vals;
  auto north_mass_flux_x = top_mass_flux_x_.host_vals;
  auto north_mass_flux_y = top_mass_flux_y_.host_vals;

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel density0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { density0(j, y_hi + 2 + k) = north_density0(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel density1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { density1(j, y_hi + 2 + k) = north_density1(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel energy0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { energy0(j, y_hi + 2 + k) = north_energy0(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel spec_energy1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { spec_energy1(j, y_hi + 2 + k) = north_energy1(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel cell_pressure", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { cell_pressure(j, y_hi + 2 + k) = north_pressure(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel shock_damping", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { shock_damping(j, y_hi + 2 + k) = north_viscosity(j, north_ylo - 1 + 2 + k); });
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel wave_speed", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { wave_speed(j, y_hi + 2 + k) = north_sound_speed(j, north_ylo - 1 + 2 + k); });
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel velx0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { velx0(j, y_hi + 1 + 2 + k) = north_velx0(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel velx1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { velx1(j, y_hi + 1 + 2 + k) = north_velx1(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel vely0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { vely0(j, y_hi + 1 + 2 + k) = north_vely0(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel vely1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { vely1(j, y_hi + 1 + 2 + k) = north_vely1(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel volume_flux_x", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { volume_flux_x(j, y_hi + 2 + k) = north_volume_flux_x(j, north_ylo - 1 + 2 + k); });
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel face_mass_flux_x", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { face_mass_flux_x(j, y_hi + 2 + k) = north_mass_flux_x(j, north_ylo - 1 + 2 + k); });
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel volume_flux_y", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { volume_flux_y(j, y_hi + 1 + 2 + k) = north_volume_flux_y(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel face_mass_flux_y", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { face_mass_flux_y(j, y_hi + 1 + 2 + k) = north_mass_flux_y(j, north_ylo + 1 - 1 + 2 + k); });
    }
  }
}

void copy_tile_ghosts_south_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0_,
    hydro2d::DeviceArray2D<double> &energy0_, hydro2d::DeviceArray2D<double> &pressure_, hydro2d::DeviceArray2D<double> &viscosity_,
    hydro2d::DeviceArray2D<double> &soundspeed_, hydro2d::DeviceArray2D<double> &density1_, hydro2d::DeviceArray2D<double> &energy1_,
    hydro2d::DeviceArray2D<double> &xvel0_, hydro2d::DeviceArray2D<double> &yvel0_, hydro2d::DeviceArray2D<double> &xvel1_, hydro2d::DeviceArray2D<double> &yvel1_,
    hydro2d::DeviceArray2D<double> &vol_flux_x_, hydro2d::DeviceArray2D<double> &vol_flux_y_, hydro2d::DeviceArray2D<double> &mass_flux_x_,
    hydro2d::DeviceArray2D<double> &mass_flux_y_, int south_xlo, int south_xhi, int south_ylo, int south_yhi,
    hydro2d::DeviceArray2D<double> &bottom_density0_, hydro2d::DeviceArray2D<double> &bottom_energy0_, hydro2d::DeviceArray2D<double> &bottom_pressure_,
    hydro2d::DeviceArray2D<double> &bottom_viscosity_, hydro2d::DeviceArray2D<double> &bottom_soundspeed_, hydro2d::DeviceArray2D<double> &bottom_density1_,
    hydro2d::DeviceArray2D<double> &bottom_energy1_, hydro2d::DeviceArray2D<double> &bottom_xvel0_, hydro2d::DeviceArray2D<double> &bottom_yvel0_,
    hydro2d::DeviceArray2D<double> &bottom_xvel1_, hydro2d::DeviceArray2D<double> &bottom_yvel1_, hydro2d::DeviceArray2D<double> &bottom_vol_flux_x_,
    hydro2d::DeviceArray2D<double> &bottom_vol_flux_y_, hydro2d::DeviceArray2D<double> &bottom_mass_flux_x_,
    hydro2d::DeviceArray2D<double> &bottom_mass_flux_y_, const int exchange_mask[FIELD_COUNT], int halo_width) {

  auto density0 = density0_.host_vals;
  auto energy0 = energy0_.host_vals;
  auto cell_pressure = pressure_.host_vals;
  auto shock_damping = viscosity_.host_vals;
  auto wave_speed = soundspeed_.host_vals;
  auto density1 = density1_.host_vals;
  auto spec_energy1 = energy1_.host_vals;
  auto velx0 = xvel0_.host_vals;
  auto vely0 = yvel0_.host_vals;
  auto velx1 = xvel1_.host_vals;
  auto vely1 = yvel1_.host_vals;
  auto volume_flux_x = vol_flux_x_.host_vals;
  auto volume_flux_y = vol_flux_y_.host_vals;
  auto face_mass_flux_x = mass_flux_x_.host_vals;
  auto face_mass_flux_y = mass_flux_y_.host_vals;
  auto south_density0 = bottom_density0_.host_vals;
  auto south_energy0 = bottom_energy0_.host_vals;
  auto south_pressure = bottom_pressure_.host_vals;
  auto south_viscosity = bottom_viscosity_.host_vals;
  auto south_sound_speed = bottom_soundspeed_.host_vals;
  auto south_density1 = bottom_density1_.host_vals;
  auto south_energy1 = bottom_energy1_.host_vals;
  auto south_velx0 = bottom_xvel0_.host_vals;
  auto south_vely0 = bottom_yvel0_.host_vals;
  auto south_velx1 = bottom_xvel1_.host_vals;
  auto south_vely1 = bottom_yvel1_.host_vals;
  auto south_volume_flux_x = bottom_vol_flux_x_.host_vals;
  auto south_volume_flux_y = bottom_vol_flux_y_.host_vals;
  auto south_mass_flux_x = bottom_mass_flux_x_.host_vals;
  auto south_mass_flux_y = bottom_mass_flux_y_.host_vals;

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel density0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { density0(j, y_lo - k) = south_density0(j, south_yhi + 1 - k); });
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel density1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { density1(j, y_lo - k) = south_density1(j, south_yhi + 1 - k); });
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel energy0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { energy0(j, y_lo - k) = south_energy0(j, south_yhi + 1 - k); });
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel spec_energy1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { spec_energy1(j, y_lo - k) = south_energy1(j, south_yhi + 1 - k); });
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel cell_pressure", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { cell_pressure(j, y_lo - k) = south_pressure(j, south_yhi + 1 - k); });
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel shock_damping", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { shock_damping(j, y_lo - k) = south_viscosity(j, south_yhi + 1 - k); });
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel wave_speed", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { wave_speed(j, y_lo - k) = south_sound_speed(j, south_yhi + 1 - k); });
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel velx0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { velx0(j, y_lo - k) = south_velx0(j, south_yhi + 1 - k); });
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel velx1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { velx1(j, y_lo - k) = south_velx1(j, south_yhi + 1 - k); });
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel vely0", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { vely0(j, y_lo - k) = south_vely0(j, south_yhi + 1 - k); });
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel vely1", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { vely1(j, y_lo - k) = south_vely1(j, south_yhi + 1 - k); });
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel volume_flux_x", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { volume_flux_x(j, y_lo - k) = south_volume_flux_x(j, south_yhi + 1 - k); });
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel face_mass_flux_x", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + 1 + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { face_mass_flux_x(j, y_lo - k) = south_mass_flux_x(j, south_yhi + 1 - k); });
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel volume_flux_y", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { volume_flux_y(j, y_lo - k) = south_volume_flux_y(j, south_yhi + 1 - k); });
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      Kokkos::parallel_for(
          "copy_tile_ghosts_north_kernel face_mass_flux_y", Kokkos::RangePolicy<>(x_lo - halo_width + 1, x_hi + halo_width + 2),
          KOKKOS_LAMBDA(const int j) { face_mass_flux_y(j, y_lo - k) = south_mass_flux_y(j, south_yhi + 1 - k); });
    }
  }
}
