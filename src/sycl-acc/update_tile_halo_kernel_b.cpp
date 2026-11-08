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

void copy_tile_ghosts_south_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &local_density0_buf,
    hydro2d::DeviceArray2D<double> &local_energy0_buf, hydro2d::DeviceArray2D<double> &local_pressure_buf, hydro2d::DeviceArray2D<double> &local_viscosity_buf,
    hydro2d::DeviceArray2D<double> &local_soundspeed_buf, hydro2d::DeviceArray2D<double> &local_density1_buf, hydro2d::DeviceArray2D<double> &local_energy1_buf,
    hydro2d::DeviceArray2D<double> &local_velx0_buf, hydro2d::DeviceArray2D<double> &local_vely0_buf, hydro2d::DeviceArray2D<double> &local_velx1_buf,
    hydro2d::DeviceArray2D<double> &local_vely1_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf,
    hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf, int south_xlo, int south_xhi,
    int south_ylo, int south_yhi, hydro2d::DeviceArray2D<double> &south_density0_buf, hydro2d::DeviceArray2D<double> &south_energy0_buf,
    hydro2d::DeviceArray2D<double> &south_pressure_buf, hydro2d::DeviceArray2D<double> &south_viscosity_buf,
    hydro2d::DeviceArray2D<double> &south_soundspeed_buf, hydro2d::DeviceArray2D<double> &south_density1_buf,
    hydro2d::DeviceArray2D<double> &south_energy1_buf, hydro2d::DeviceArray2D<double> &south_velx0_buf,
    hydro2d::DeviceArray2D<double> &south_vely0_buf, hydro2d::DeviceArray2D<double> &south_velx1_buf,
    hydro2d::DeviceArray2D<double> &south_vely1_buf, hydro2d::DeviceArray2D<double> &south_volume_flux_x_buf,
    hydro2d::DeviceArray2D<double> &south_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &south_mass_flux_x_buf,
    hydro2d::DeviceArray2D<double> &south_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_density0 = south_density0_buf.access<R>(h);
        auto density0 = local_density0_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_density0>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          density0[j[0]][y_lo - k] = south_density0[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_density1 = south_density1_buf.access<R>(h);
        auto density1 = local_density1_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_density1>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          density1[j[0]][y_lo - k] = south_density1[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_energy0 = south_energy0_buf.access<R>(h);
        auto energy0 = local_energy0_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_energy0>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          energy0[j[0]][y_lo - k] = south_energy0[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_energy1 = south_energy1_buf.access<R>(h);
        auto spec_energy1 = local_energy1_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_energy1>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          spec_energy1[j[0]][y_lo - k] = south_energy1[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_pressure = south_pressure_buf.access<R>(h);
        auto cell_pressure = local_pressure_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_pressure>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          cell_pressure[j[0]][y_lo - k] = south_pressure[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Viscocity
  if (exchange_mask[fld_viscosity] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_viscosity = south_viscosity_buf.access<R>(h);
        auto shock_damping = local_viscosity_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_viscosity>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          shock_damping[j[0]][y_lo - k] = south_viscosity[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      //  DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_sound_speed = south_soundspeed_buf.access<R>(h);
        auto wave_speed = local_soundspeed_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_soundspeed>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          wave_speed[j[0]][y_lo - k] = south_sound_speed[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_velx0 = south_velx0_buf.access<R>(h);
        auto velx0 = local_velx0_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_xvel0>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                                                   [=](id<1> j) { velx0[j[0]][y_lo - k] = south_velx0[j[0]][south_yhi + 1 - k]; });
      });
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_velx1 = south_velx1_buf.access<R>(h);
        auto velx1 = local_velx1_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_xvel1>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                                                   [=](id<1> j) { velx1[j[0]][y_lo - k] = south_velx1[j[0]][south_yhi + 1 - k]; });
      });
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_vely0 = south_vely0_buf.access<R>(h);
        auto vely0 = local_vely0_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_yvel0>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                                                   [=](id<1> j) { vely0[j[0]][y_lo - k] = south_vely0[j[0]][south_yhi + 1 - k]; });
      });
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_vely1 = south_vely1_buf.access<R>(h);
        auto vely1 = local_vely1_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_yvel1>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2},
                                                   [=](id<1> j) { vely1[j[0]][y_lo - k] = south_vely1[j[0]][south_yhi + 1 - k]; });
      });
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_volume_flux_x = south_volume_flux_x_buf.access<R>(h);
        auto volume_flux_x = local_volume_flux_x_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_vol_flux_x>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](id<1> j) {
          volume_flux_x[j[0]][y_lo - k] = south_volume_flux_x[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+1+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_mass_flux_x = south_mass_flux_x_buf.access<R>(h);
        auto face_mass_flux_x = local_mass_flux_x_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_mass_flux_x>(h, {x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](id<1> j) {
          face_mass_flux_x[j[0]][y_lo - k] = south_mass_flux_x[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_volume_flux_y = south_volume_flux_y_buf.access<R>(h);
        auto volume_flux_y = local_volume_flux_y_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_vol_flux_y>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          volume_flux_y[j[0]][y_lo - k] = south_volume_flux_y[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    for (int k = 0; k < halo_width; ++k) {
      // DO j=x_lo-halo_width, x_hi+halo_width
      hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
        auto south_mass_flux_y = south_mass_flux_y_buf.access<R>(h);
        auto face_mass_flux_y = local_mass_flux_y_buf.access<W>(h);
        hydro2d::par_ranged<class upd_halo_b_mass_flux_y>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          face_mass_flux_y[j[0]][y_lo - k] = south_mass_flux_y[j[0]][south_yhi + 1 - k];
        });
      });
    }
  }
}
