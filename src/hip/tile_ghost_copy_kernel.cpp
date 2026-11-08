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
#include "context.h"

//   @brief Fortran kernel to update the external halo cell_total in a subdomain.
//   @author Wayne Gaudin
//   @details Updates halo cell_total for the required exchange_mask at the required halo_width
//   for any halo cell_total copied lie on an external boundary. The origin_site and type
//   of data governs how this is carried out. External boundaries are always
//   reflective.

void copy_tile_ghosts_west_kernel(
    sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0, hydro2d::DeviceArray2D<double> &energy0,
    hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping, hydro2d::DeviceArray2D<double> &wave_speed,
    hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1, hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0,
    hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x,
    hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int west_xlo,
    int west_xhi, int west_ylo, int west_yhi, hydro2d::DeviceArray2D<double> &west_density0, hydro2d::DeviceArray2D<double> &west_energy0,
    hydro2d::DeviceArray2D<double> &west_pressure, hydro2d::DeviceArray2D<double> &west_viscosity, hydro2d::DeviceArray2D<double> &west_sound_speed,
    hydro2d::DeviceArray2D<double> &west_density1, hydro2d::DeviceArray2D<double> &west_energy1, hydro2d::DeviceArray2D<double> &west_velx0,
    hydro2d::DeviceArray2D<double> &west_vely0, hydro2d::DeviceArray2D<double> &west_velx1, hydro2d::DeviceArray2D<double> &west_vely1,
    hydro2d::DeviceArray2D<double> &west_volume_flux_x, hydro2d::DeviceArray2D<double> &west_volume_flux_y, hydro2d::DeviceArray2D<double> &west_mass_flux_x,
    hydro2d::DeviceArray2D<double> &west_mass_flux_y, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density0(x_lo - j, k) = west_density0(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density1(x_lo - j, k) = west_density1(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            energy0(x_lo - j, k) = west_energy0(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            spec_energy1(x_lo - j, k) = west_energy1(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            cell_pressure(x_lo - j, k) = west_pressure(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            shock_damping(x_lo - j, k) = west_viscosity(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            wave_speed(x_lo - j, k) = west_sound_speed(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx0(x_lo - j, k) = west_velx0(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx1(x_lo - j, k) = west_velx1(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely0(x_lo - j, k) = west_vely0(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely1(x_lo - j, k) = west_vely1(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_x(x_lo - j, k) = west_volume_flux_x(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_x(x_lo - j, k) = west_mass_flux_x(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_y(x_lo - j, k) = west_volume_flux_y(west_xhi + 1 - j, k);
                          }
                        }));
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_y(x_lo - j, k) = west_mass_flux_y(west_xhi + 1 - j, k);
                          }
                        }));
  }
}

void copy_tile_ghosts_east_kernel(
    sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0, hydro2d::DeviceArray2D<double> &energy0,
    hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping, hydro2d::DeviceArray2D<double> &wave_speed,
    hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1, hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0,
    hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x,
    hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int east_xlo,
    int east_xhi, int east_ylo, int east_yhi, hydro2d::DeviceArray2D<double> &east_density0, hydro2d::DeviceArray2D<double> &east_energy0,
    hydro2d::DeviceArray2D<double> &east_pressure, hydro2d::DeviceArray2D<double> &east_viscosity, hydro2d::DeviceArray2D<double> &east_sound_speed,
    hydro2d::DeviceArray2D<double> &east_density1, hydro2d::DeviceArray2D<double> &east_energy1, hydro2d::DeviceArray2D<double> &east_velx0,
    hydro2d::DeviceArray2D<double> &east_vely0, hydro2d::DeviceArray2D<double> &east_velx1, hydro2d::DeviceArray2D<double> &east_vely1,
    hydro2d::DeviceArray2D<double> &east_volume_flux_x, hydro2d::DeviceArray2D<double> &east_volume_flux_y, hydro2d::DeviceArray2D<double> &east_mass_flux_x,
    hydro2d::DeviceArray2D<double> &east_mass_flux_y, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density0(x_hi + 2 + j, k) = east_density0(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density1(x_hi + 2 + j, k) = east_density1(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            energy0(x_hi + 2 + j, k) = east_energy0(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            spec_energy1(x_hi + 2 + j, k) = east_energy1(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            cell_pressure(x_hi + 2 + j, k) = east_pressure(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            shock_damping(x_hi + 2 + j, k) = east_viscosity(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            wave_speed(x_hi + 2 + j, k) = east_sound_speed(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx0(x_hi + 1 + 2 + j, k) = east_velx0(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx1(x_hi + 1 + 2 + j, k) = east_velx1(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely0(x_hi + 1 + 2 + j, k) = east_vely0(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely1(x_hi + 1 + 2 + j, k) = east_vely1(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_x(x_hi + 1 + 2 + j, k) = east_volume_flux_x(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_x(x_hi + 1 + 2 + j, k) = east_mass_flux_x(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_y(x_hi + 2 + j, k) = east_volume_flux_y(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=] DEVICE_KERNEL(int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_y(x_hi + 2 + j, k) = east_mass_flux_y(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }
}

//  Top and first_row only do x_lower -> x_upper
//  This is because the corner ghosts will get communicated in the first_col last_col
//  communication

void copy_tile_ghosts_north_kernel(sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
                               hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
                               hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
                               hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1,
                               hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y,
                               hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int north_xlo, int north_xhi,
                               int north_ylo, int north_yhi, hydro2d::DeviceArray2D<double> &north_density0, hydro2d::DeviceArray2D<double> &north_energy0,
                               hydro2d::DeviceArray2D<double> &north_pressure, hydro2d::DeviceArray2D<double> &north_viscosity,
                               hydro2d::DeviceArray2D<double> &north_sound_speed, hydro2d::DeviceArray2D<double> &north_density1,
                               hydro2d::DeviceArray2D<double> &north_energy1, hydro2d::DeviceArray2D<double> &north_velx0,
                               hydro2d::DeviceArray2D<double> &north_vely0, hydro2d::DeviceArray2D<double> &north_velx1,
                               hydro2d::DeviceArray2D<double> &north_vely1, hydro2d::DeviceArray2D<double> &north_volume_flux_x,
                               hydro2d::DeviceArray2D<double> &north_volume_flux_y, hydro2d::DeviceArray2D<double> &north_mass_flux_x,
                               hydro2d::DeviceArray2D<double> &north_mass_flux_y, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { density0(j, y_hi + 2 + k) = north_density0(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { density1(j, y_hi + 2 + k) = north_density1(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { energy0(j, y_hi + 2 + k) = north_energy0(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { spec_energy1(j, y_hi + 2 + k) = north_energy1(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { cell_pressure(j, y_hi + 2 + k) = north_pressure(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { shock_damping(j, y_hi + 2 + k) = north_viscosity(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { wave_speed(j, y_hi + 2 + k) = north_sound_speed(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { velx0(j, y_hi + 1 + 2 + k) = north_velx0(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { velx1(j, y_hi + 1 + 2 + k) = north_velx1(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { vely0(j, y_hi + 1 + 2 + k) = north_vely0(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { vely1(j, y_hi + 1 + 2 + k) = north_vely1(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { volume_flux_x(j, y_hi + 2 + k) = north_volume_flux_x(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { face_mass_flux_x(j, y_hi + 2 + k) = north_mass_flux_x(j, north_ylo - 1 + 2 + k); }));
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { volume_flux_y(j, y_hi + 1 + 2 + k) = north_volume_flux_y(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { face_mass_flux_y(j, y_hi + 1 + 2 + k) = north_mass_flux_y(j, north_ylo + 1 - 1 + 2 + k); }));
    }
  }
}

void copy_tile_ghosts_south_kernel(
    sim_state &, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0, hydro2d::DeviceArray2D<double> &energy0,
    hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping, hydro2d::DeviceArray2D<double> &wave_speed,
    hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1, hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0,
    hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x,
    hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int south_xlo,
    int south_xhi, int south_ylo, int south_yhi, hydro2d::DeviceArray2D<double> &south_density0, hydro2d::DeviceArray2D<double> &south_energy0,
    hydro2d::DeviceArray2D<double> &south_pressure, hydro2d::DeviceArray2D<double> &south_viscosity, hydro2d::DeviceArray2D<double> &south_sound_speed,
    hydro2d::DeviceArray2D<double> &south_density1, hydro2d::DeviceArray2D<double> &south_energy1, hydro2d::DeviceArray2D<double> &south_velx0,
    hydro2d::DeviceArray2D<double> &south_vely0, hydro2d::DeviceArray2D<double> &south_velx1, hydro2d::DeviceArray2D<double> &south_vely1,
    hydro2d::DeviceArray2D<double> &south_volume_flux_x, hydro2d::DeviceArray2D<double> &south_volume_flux_y, hydro2d::DeviceArray2D<double> &south_mass_flux_x,
    hydro2d::DeviceArray2D<double> &south_mass_flux_y, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { density0(j, y_lo - k) = south_density0(j, south_yhi + 1 - k); }));
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { density1(j, y_lo - k) = south_density1(j, south_yhi + 1 - k); }));
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { energy0(j, y_lo - k) = south_energy0(j, south_yhi + 1 - k); }));
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { spec_energy1(j, y_lo - k) = south_energy1(j, south_yhi + 1 - k); }));
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { cell_pressure(j, y_lo - k) = south_pressure(j, south_yhi + 1 - k); }));
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { shock_damping(j, y_lo - k) = south_viscosity(j, south_yhi + 1 - k); }));
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { wave_speed(j, y_lo - k) = south_sound_speed(j, south_yhi + 1 - k); }));
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { velx0(j, y_lo - k) = south_velx0(j, south_yhi + 1 - k); }));
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { velx1(j, y_lo - k) = south_velx1(j, south_yhi + 1 - k); }));
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { vely0(j, y_lo - k) = south_vely0(j, south_yhi + 1 - k); }));
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { vely1(j, y_lo - k) = south_vely1(j, south_yhi + 1 - k); }));
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { volume_flux_x(j, y_lo - k) = south_volume_flux_x(j, south_yhi + 1 - k); }));
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { face_mass_flux_x(j, y_lo - k) = south_mass_flux_x(j, south_yhi + 1 - k); }));
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { volume_flux_y(j, y_lo - k) = south_volume_flux_y(j, south_yhi + 1 - k); }));
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=] DEVICE_KERNEL(int j) { face_mass_flux_y(j, y_lo - k) = south_mass_flux_y(j, south_yhi + 1 - k); }));
    }
  }
}
