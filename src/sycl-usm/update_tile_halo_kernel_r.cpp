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

#include "context.h"
#include "tile_ghost_copy_kernel.h"

void copy_tile_ghosts_east_kernel( //
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
    hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
    hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
    hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1,
    hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &face_mass_flux_x,
    hydro2d::DeviceArray2D<double> &face_mass_flux_y, int east_xlo, int east_xhi, int east_ylo, int east_yhi,
    hydro2d::DeviceArray2D<double> &east_density0, hydro2d::DeviceArray2D<double> &east_energy0, hydro2d::DeviceArray2D<double> &east_pressure,
    hydro2d::DeviceArray2D<double> &east_viscosity, hydro2d::DeviceArray2D<double> &east_sound_speed, hydro2d::DeviceArray2D<double> &east_density1,
    hydro2d::DeviceArray2D<double> &east_energy1, hydro2d::DeviceArray2D<double> &east_velx0, hydro2d::DeviceArray2D<double> &east_vely0,
    hydro2d::DeviceArray2D<double> &east_velx1, hydro2d::DeviceArray2D<double> &east_vely1, hydro2d::DeviceArray2D<double> &east_volume_flux_x,
    hydro2d::DeviceArray2D<double> &east_volume_flux_y, hydro2d::DeviceArray2D<double> &east_mass_flux_x, hydro2d::DeviceArray2D<double> &east_mass_flux_y,
    const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density0(x_hi + 2 + j, k) = east_density0(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            density1(x_hi + 2 + j, k) = east_density1(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            energy0(x_hi + 2 + j, k) = east_energy0(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            spec_energy1(x_hi + 2 + j, k) = east_energy1(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            cell_pressure(x_hi + 2 + j, k) = east_pressure(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            shock_damping(x_hi + 2 + j, k) = east_viscosity(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            wave_speed(x_hi + 2 + j, k) = east_sound_speed(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx0(x_hi + 1 + 2 + j, k) = east_velx0(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            velx1(x_hi + 1 + 2 + j, k) = east_velx1(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely0(x_hi + 1 + 2 + j, k) = east_vely0(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            vely1(x_hi + 1 + 2 + j, k) = east_vely1(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_x(x_hi + 1 + 2 + j, k) = east_volume_flux_x(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_x(x_hi + 1 + 2 + j, k) = east_mass_flux_x(east_xlo + 1 - 1 + 2 + j, k);
                          }
                        }));
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            volume_flux_y(x_hi + 2 + j, k) = east_volume_flux_y(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, ([=](int k) {
                          for (int j = 0; j < halo_width; ++j) {
                            face_mass_flux_y(x_hi + 2 + j, k) = east_mass_flux_y(east_xlo - 1 + 2 + j, k);
                          }
                        }));
  }
}
