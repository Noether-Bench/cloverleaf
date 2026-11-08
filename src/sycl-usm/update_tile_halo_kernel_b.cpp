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

void copy_tile_ghosts_south_kernel( //
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
    hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
    hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
    hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1, hydro2d::DeviceArray2D<double> &vely1,
    hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &face_mass_flux_x,
    hydro2d::DeviceArray2D<double> &face_mass_flux_y, int south_xlo, int south_xhi, int south_ylo, int south_yhi,
    hydro2d::DeviceArray2D<double> &south_density0, hydro2d::DeviceArray2D<double> &south_energy0, hydro2d::DeviceArray2D<double> &south_pressure,
    hydro2d::DeviceArray2D<double> &south_viscosity, hydro2d::DeviceArray2D<double> &south_sound_speed, hydro2d::DeviceArray2D<double> &south_density1,
    hydro2d::DeviceArray2D<double> &south_energy1, hydro2d::DeviceArray2D<double> &south_velx0, hydro2d::DeviceArray2D<double> &south_vely0,
    hydro2d::DeviceArray2D<double> &south_velx1, hydro2d::DeviceArray2D<double> &south_vely1, hydro2d::DeviceArray2D<double> &south_volume_flux_x,
    hydro2d::DeviceArray2D<double> &south_volume_flux_y, hydro2d::DeviceArray2D<double> &south_mass_flux_x, hydro2d::DeviceArray2D<double> &south_mass_flux_y,
    const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { density0(j, y_lo - k) = south_density0(j, south_yhi + 1 - k); }));
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { density1(j, y_lo - k) = south_density1(j, south_yhi + 1 - k); }));
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { energy0(j, y_lo - k) = south_energy0(j, south_yhi + 1 - k); }));
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { spec_energy1(j, y_lo - k) = south_energy1(j, south_yhi + 1 - k); }));
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { cell_pressure(j, y_lo - k) = south_pressure(j, south_yhi + 1 - k); }));
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { shock_damping(j, y_lo - k) = south_viscosity(j, south_yhi + 1 - k); }));
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { wave_speed(j, y_lo - k) = south_sound_speed(j, south_yhi + 1 - k); }));
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { velx0(j, y_lo - k) = south_velx0(j, south_yhi + 1 - k); }));
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { velx1(j, y_lo - k) = south_velx1(j, south_yhi + 1 - k); }));
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { vely0(j, y_lo - k) = south_vely0(j, south_yhi + 1 - k); }));
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { vely1(j, y_lo - k) = south_vely1(j, south_yhi + 1 - k); }));
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { volume_flux_x(j, y_lo - k) = south_volume_flux_x(j, south_yhi + 1 - k); }));
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                          ([=](int j) { face_mass_flux_x(j, y_lo - k) = south_mass_flux_x(j, south_yhi + 1 - k); }));
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { volume_flux_y(j, y_lo - k) = south_volume_flux_y(j, south_yhi + 1 - k); }));
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width

      hydro2d::par_ranged1(globals.context.exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2},
                          ([=](int j) { face_mass_flux_y(j, y_lo - k) = south_mass_flux_y(j, south_yhi + 1 - k); }));
    }
  }
}
