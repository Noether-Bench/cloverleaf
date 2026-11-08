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

void copy_tile_ghosts_east_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               hydro2d::DeviceArray2D<double> &local_density0_buf, hydro2d::DeviceArray2D<double> &local_energy0_buf,
                               hydro2d::DeviceArray2D<double> &local_pressure_buf, hydro2d::DeviceArray2D<double> &local_viscosity_buf,
                               hydro2d::DeviceArray2D<double> &local_soundspeed_buf, hydro2d::DeviceArray2D<double> &local_density1_buf,
                               hydro2d::DeviceArray2D<double> &local_energy1_buf, hydro2d::DeviceArray2D<double> &local_velx0_buf,
                               hydro2d::DeviceArray2D<double> &local_vely0_buf, hydro2d::DeviceArray2D<double> &local_velx1_buf,
                               hydro2d::DeviceArray2D<double> &local_vely1_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf, int east_xlo, int east_xhi, int east_ylo, int east_yhi,
                               hydro2d::DeviceArray2D<double> &east_density0_buf, hydro2d::DeviceArray2D<double> &east_energy0_buf,
                               hydro2d::DeviceArray2D<double> &east_pressure_buf, hydro2d::DeviceArray2D<double> &east_viscosity_buf,
                               hydro2d::DeviceArray2D<double> &east_soundspeed_buf, hydro2d::DeviceArray2D<double> &east_density1_buf,
                               hydro2d::DeviceArray2D<double> &east_energy1_buf, hydro2d::DeviceArray2D<double> &east_velx0_buf,
                               hydro2d::DeviceArray2D<double> &east_vely0_buf, hydro2d::DeviceArray2D<double> &east_velx1_buf,
                               hydro2d::DeviceArray2D<double> &east_vely1_buf, hydro2d::DeviceArray2D<double> &east_volume_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &east_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &east_mass_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &east_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {

  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_density0 = east_density0_buf.access<R>(h);
      auto density0 = local_density0_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_density0>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          density0[x_hi + 2 + j][k[0]] = east_density0[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_density1 = east_density1_buf.access<R>(h);
      auto density1 = local_density1_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_density1>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          density1[x_hi + 2 + j][k[0]] = east_density1[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_energy0 = east_energy0_buf.access<R>(h);
      auto energy0 = local_energy0_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_energy0>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          energy0[x_hi + 2 + j][k[0]] = east_energy0[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_energy1 = east_energy1_buf.access<R>(h);
      auto spec_energy1 = local_energy1_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_energy1>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          spec_energy1[x_hi + 2 + j][k[0]] = east_energy1[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto cell_pressure = local_pressure_buf.access<W>(h);
      auto east_pressure = east_pressure_buf.access<R>(h);
      hydro2d::par_ranged<class upd_halo_r_pressure>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          cell_pressure[x_hi + 2 + j][k[0]] = east_pressure[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_viscosity = east_viscosity_buf.access<R>(h);
      auto shock_damping = local_viscosity_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_viscosity>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          shock_damping[x_hi + 2 + j][k[0]] = east_viscosity[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_sound_speed = east_soundspeed_buf.access<R>(h);
      auto wave_speed = local_soundspeed_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_soundspeed>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          wave_speed[x_hi + 2 + j][k[0]] = east_sound_speed[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_velx0 = east_velx0_buf.access<R>(h);
      auto velx0 = local_velx0_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_xvel0>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          velx0[x_hi + 1 + 2 + j][k[0]] = east_velx0[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_velx1 = east_velx1_buf.access<R>(h);
      auto velx1 = local_velx1_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_xvel1>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          velx1[x_hi + 1 + 2 + j][k[0]] = east_velx1[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_vely0 = east_vely0_buf.access<R>(h);
      auto vely0 = local_vely0_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_yvel0>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          vely0[x_hi + 1 + 2 + j][k[0]] = east_vely0[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_vely1 = east_vely1_buf.access<R>(h);
      auto vely1 = local_vely1_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_yvel1>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          vely1[x_hi + 1 + 2 + j][k[0]] = east_vely1[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_volume_flux_x = east_volume_flux_x_buf.access<R>(h);
      auto volume_flux_x = local_volume_flux_x_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_vol_flux_x>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_x[x_hi + 1 + 2 + j][k[0]] = east_volume_flux_x[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_mass_flux_x = east_mass_flux_x_buf.access<R>(h);
      auto face_mass_flux_x = local_mass_flux_x_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_mass_flux_x>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_x[x_hi + 1 + 2 + j][k[0]] = east_mass_flux_x[east_xlo + 1 - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_volume_flux_y = east_volume_flux_y_buf.access<R>(h);
      auto volume_flux_y = local_volume_flux_y_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_vol_flux_y>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_y[x_hi + 2 + j][k[0]] = east_volume_flux_y[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width
    hydro2d::execute(globals.context.exec_ctx, [&](handler &h) {
      auto east_mass_flux_y = east_mass_flux_y_buf.access<R>(h);
      auto face_mass_flux_y = local_mass_flux_y_buf.access<W>(h);
      hydro2d::par_ranged<class upd_halo_r_mass_flux_y>(h, {y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](id<1> k) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_y[x_hi + 2 + j][k[0]] = east_mass_flux_y[east_xlo - 1 + 2 + j][k[0]];
        }
      });
    });
  }
}
