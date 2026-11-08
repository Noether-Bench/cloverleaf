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

void copy_tile_ghosts_west_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               hydro2d::DeviceArray2D<double> &local_density0_buf, hydro2d::DeviceArray2D<double> &local_energy0_buf,
                               hydro2d::DeviceArray2D<double> &local_pressure_buf, hydro2d::DeviceArray2D<double> &local_viscosity_buf,
                               hydro2d::DeviceArray2D<double> &local_soundspeed_buf, hydro2d::DeviceArray2D<double> &local_density1_buf,
                               hydro2d::DeviceArray2D<double> &local_energy1_buf, hydro2d::DeviceArray2D<double> &local_velx0_buf,
                               hydro2d::DeviceArray2D<double> &local_vely0_buf, hydro2d::DeviceArray2D<double> &local_velx1_buf,
                               hydro2d::DeviceArray2D<double> &local_vely1_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf, int west_xlo, int west_xhi, int west_ylo, int west_yhi,
                               hydro2d::DeviceArray2D<double> &west_density0_buf, hydro2d::DeviceArray2D<double> &west_energy0_buf,
                               hydro2d::DeviceArray2D<double> &west_pressure_buf, hydro2d::DeviceArray2D<double> &west_viscosity_buf,
                               hydro2d::DeviceArray2D<double> &west_soundspeed_buf, hydro2d::DeviceArray2D<double> &west_density1_buf,
                               hydro2d::DeviceArray2D<double> &west_energy1_buf, hydro2d::DeviceArray2D<double> &west_velx0_buf,
                               hydro2d::DeviceArray2D<double> &west_vely0_buf, hydro2d::DeviceArray2D<double> &west_velx1_buf,
                               hydro2d::DeviceArray2D<double> &west_vely1_buf, hydro2d::DeviceArray2D<double> &west_volume_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &west_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &west_mass_flux_x_buf,
                               hydro2d::DeviceArray2D<double> &west_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *west_density0 = west_density0_buf.data;
    size_t west_density0_nx = west_density0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : local_density0_buf.N()], west_density0[ : west_density0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density0[(x_lo - j) + (k)*local_density0_nx] = west_density0[(west_xhi + 1 - j) + (k)*west_density0_nx];
      }
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *west_density1 = west_density1_buf.data;
    size_t west_density1_nx = west_density1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : local_density1_buf.N()], west_density1[ : west_density1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density1[(x_lo - j) + (k)*local_density1_nx] = west_density1[(west_xhi + 1 - j) + (k)*west_density1_nx];
      }
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *west_energy0 = west_energy0_buf.data;
    size_t west_energy0_nx = west_energy0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : local_energy0_buf.N()], west_energy0[ : west_energy0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        energy0[(x_lo - j) + (k)*local_energy0_nx] = west_energy0[(west_xhi + 1 - j) + (k)*west_energy0_nx];
      }
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *west_energy1 = west_energy1_buf.data;
    size_t west_energy1_nx = west_energy1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(spec_energy1[ : local_energy1_buf.N()], west_energy1[ : west_energy1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        spec_energy1[(x_lo - j) + (k)*local_energy1_nx] = west_energy1[(west_xhi + 1 - j) + (k)*west_energy1_nx];
      }
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *west_pressure = west_pressure_buf.data;
    size_t west_pressure_nx = west_pressure_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(cell_pressure[ : local_pressure_buf.N()], west_pressure[ : west_pressure_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        cell_pressure[(x_lo - j) + (k)*local_pressure_nx] = west_pressure[(west_xhi + 1 - j) + (k)*west_pressure_nx];
      }
    }
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *west_viscosity = west_viscosity_buf.data;
    size_t west_viscosity_nx = west_viscosity_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(shock_damping[ : local_viscosity_buf.N()], west_viscosity[ : west_viscosity_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        shock_damping[(x_lo - j) + (k)*local_viscosity_nx] = west_viscosity[(west_xhi + 1 - j) + (k)*west_viscosity_nx];
      }
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *west_sound_speed = west_soundspeed_buf.data;
    size_t west_soundspeed_nx = west_soundspeed_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(wave_speed[ : local_soundspeed_buf.N()], west_sound_speed[ : west_soundspeed_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        wave_speed[(x_lo - j) + (k)*local_soundspeed_nx] = west_sound_speed[(west_xhi + 1 - j) + (k)*west_soundspeed_nx];
      }
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *west_velx0 = west_velx0_buf.data;
    size_t west_velx0_nx = west_velx0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx0[ : local_velx0_buf.N()], west_velx0[ : west_velx0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx0[(x_lo - j) + (k)*local_velx0_nx] = west_velx0[(west_xhi + 1 - j) + (k)*west_velx0_nx];
      }
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *west_velx1 = west_velx1_buf.data;
    size_t west_velx1_nx = west_velx1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx1[ : local_velx1_buf.N()], west_velx1[ : west_velx1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx1[(x_lo - j) + (k)*local_velx1_nx] = west_velx1[(west_xhi + 1 - j) + (k)*west_velx1_nx];
      }
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *west_vely0 = west_vely0_buf.data;
    size_t west_vely0_nx = west_vely0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely0[ : local_vely0_buf.N()], west_vely0[ : west_vely0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely0[(x_lo - j) + (k)*local_vely0_nx] = west_vely0[(west_xhi + 1 - j) + (k)*west_vely0_nx];
      }
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *west_vely1 = west_vely1_buf.data;
    size_t west_vely1_nx = west_vely1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely1[ : local_vely1_buf.N()], west_vely1[ : west_vely1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely1[(x_lo - j) + (k)*local_vely1_nx] = west_vely1[(west_xhi + 1 - j) + (k)*west_vely1_nx];
      }
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *west_volume_flux_x = west_volume_flux_x_buf.data;
    size_t west_volume_flux_x_nx = west_volume_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_x[ : local_volume_flux_x_buf.N()], west_volume_flux_x[ : west_volume_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_x[(x_lo - j) + (k)*local_volume_flux_x_nx] = west_volume_flux_x[(west_xhi + 1 - j) + (k)*west_volume_flux_x_nx];
      }
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *west_mass_flux_x = west_mass_flux_x_buf.data;
    size_t west_mass_flux_x_nx = west_mass_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_x[ : local_mass_flux_x_buf.N()], west_mass_flux_x[ : west_mass_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_x[(x_lo - j) + (k)*local_mass_flux_x_nx] = west_mass_flux_x[(west_xhi + 1 - j) + (k)*west_mass_flux_x_nx];
      }
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *west_volume_flux_y = west_volume_flux_y_buf.data;
    size_t west_volume_flux_y_nx = west_volume_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_y[ : local_volume_flux_y_buf.N()], west_volume_flux_y[ : west_volume_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_y[(x_lo - j) + (k)*local_volume_flux_y_nx] = west_volume_flux_y[(west_xhi + 1 - j) + (k)*west_volume_flux_y_nx];
      }
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *west_mass_flux_y = west_mass_flux_y_buf.data;
    size_t west_mass_flux_y_nx = west_mass_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_y[ : local_mass_flux_y_buf.N()], west_mass_flux_y[ : west_mass_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_y[(x_lo - j) + (k)*local_mass_flux_y_nx] = west_mass_flux_y[(west_xhi + 1 - j) + (k)*west_mass_flux_y_nx];
      }
    }
  }
}

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
{
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *east_density0 = east_density0_buf.data;
    size_t east_density0_nx = east_density0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : local_density0_buf.N()], east_density0[ : east_density0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density0[(x_lo - j) + (k)*local_density0_nx] = east_density0[(east_xhi + 1 - j) + (k)*east_density0_nx];
      }
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *east_density1 = east_density1_buf.data;
    size_t east_density1_nx = east_density1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : local_density1_buf.N()], east_density1[ : east_density1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density1[(x_lo - j) + (k)*local_density1_nx] = east_density1[(east_xhi + 1 - j) + (k)*east_density1_nx];
      }
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *east_energy0 = east_energy0_buf.data;
    size_t east_energy0_nx = east_energy0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : local_energy0_buf.N()], east_energy0[ : east_energy0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        energy0[(x_lo - j) + (k)*local_energy0_nx] = east_energy0[(east_xhi + 1 - j) + (k)*east_energy0_nx];
      }
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *east_energy1 = east_energy1_buf.data;
    size_t east_energy1_nx = east_energy1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(spec_energy1[ : local_energy1_buf.N()], east_energy1[ : east_energy1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        spec_energy1[(x_lo - j) + (k)*local_energy1_nx] = east_energy1[(east_xhi + 1 - j) + (k)*east_energy1_nx];
      }
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *east_pressure = east_pressure_buf.data;
    size_t east_pressure_nx = east_pressure_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(cell_pressure[ : local_pressure_buf.N()], east_pressure[ : east_pressure_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        cell_pressure[(x_lo - j) + (k)*local_pressure_nx] = east_pressure[(east_xhi + 1 - j) + (k)*east_pressure_nx];
      }
    }
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *east_viscosity = east_viscosity_buf.data;
    size_t east_viscosity_nx = east_viscosity_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(shock_damping[ : local_viscosity_buf.N()], east_viscosity[ : east_viscosity_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        shock_damping[(x_lo - j) + (k)*local_viscosity_nx] = east_viscosity[(east_xhi + 1 - j) + (k)*east_viscosity_nx];
      }
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *east_sound_speed = east_soundspeed_buf.data;
    size_t east_soundspeed_nx = east_soundspeed_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(wave_speed[ : local_soundspeed_buf.N()], east_sound_speed[ : east_soundspeed_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        wave_speed[(x_lo - j) + (k)*local_soundspeed_nx] = east_sound_speed[(east_xhi + 1 - j) + (k)*east_soundspeed_nx];
      }
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *east_velx0 = east_velx0_buf.data;
    size_t east_velx0_nx = east_velx0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx0[ : local_velx0_buf.N()], east_velx0[ : east_velx0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx0[(x_lo - j) + (k)*local_velx0_nx] = east_velx0[(east_xhi + 1 - j) + (k)*east_velx0_nx];
      }
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *east_velx1 = east_velx1_buf.data;
    size_t east_velx1_nx = east_velx1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx1[ : local_velx1_buf.N()], east_velx1[ : east_velx1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx1[(x_lo - j) + (k)*local_velx1_nx] = east_velx1[(east_xhi + 1 - j) + (k)*east_velx1_nx];
      }
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *east_vely0 = east_vely0_buf.data;
    size_t east_vely0_nx = east_vely0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely0[ : local_vely0_buf.N()], east_vely0[ : east_vely0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely0[(x_lo - j) + (k)*local_vely0_nx] = east_vely0[(east_xhi + 1 - j) + (k)*east_vely0_nx];
      }
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *east_vely1 = east_vely1_buf.data;
    size_t east_vely1_nx = east_vely1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely1[ : local_vely1_buf.N()], east_vely1[ : east_vely1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely1[(x_lo - j) + (k)*local_vely1_nx] = east_vely1[(east_xhi + 1 - j) + (k)*east_vely1_nx];
      }
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *east_volume_flux_x = east_volume_flux_x_buf.data;
    size_t east_volume_flux_x_nx = east_volume_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_x[ : local_volume_flux_x_buf.N()], east_volume_flux_x[ : east_volume_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_x[(x_lo - j) + (k)*local_volume_flux_x_nx] = east_volume_flux_x[(east_xhi + 1 - j) + (k)*east_volume_flux_x_nx];
      }
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *east_mass_flux_x = east_mass_flux_x_buf.data;
    size_t east_mass_flux_x_nx = east_mass_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_x[ : local_mass_flux_x_buf.N()], east_mass_flux_x[ : east_mass_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_x[(x_lo - j) + (k)*local_mass_flux_x_nx] = east_mass_flux_x[(east_xhi + 1 - j) + (k)*east_mass_flux_x_nx];
      }
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *east_volume_flux_y = east_volume_flux_y_buf.data;
    size_t east_volume_flux_y_nx = east_volume_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_y[ : local_volume_flux_y_buf.N()], east_volume_flux_y[ : east_volume_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_y[(x_lo - j) + (k)*local_volume_flux_y_nx] = east_volume_flux_y[(east_xhi + 1 - j) + (k)*east_volume_flux_y_nx];
      }
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *east_mass_flux_y = east_mass_flux_y_buf.data;
    size_t east_mass_flux_y_nx = east_mass_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_y[ : local_mass_flux_y_buf.N()], east_mass_flux_y[ : east_mass_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_y[(x_lo - j) + (k)*local_mass_flux_y_nx] = east_mass_flux_y[(east_xhi + 1 - j) + (k)*east_mass_flux_y_nx];
      }
    }
  }
}}

//  Top and first_row only do x_lower -> x_upper
//  This is because the corner ghosts will get communicated in the first_col last_col
//  communication

void copy_tile_ghosts_north_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &local_density0_buf,
    hydro2d::DeviceArray2D<double> &local_energy0_buf, hydro2d::DeviceArray2D<double> &local_pressure_buf, hydro2d::DeviceArray2D<double> &local_viscosity_buf,
    hydro2d::DeviceArray2D<double> &local_soundspeed_buf, hydro2d::DeviceArray2D<double> &local_density1_buf, hydro2d::DeviceArray2D<double> &local_energy1_buf,
    hydro2d::DeviceArray2D<double> &local_velx0_buf, hydro2d::DeviceArray2D<double> &local_vely0_buf, hydro2d::DeviceArray2D<double> &local_velx1_buf,
    hydro2d::DeviceArray2D<double> &local_vely1_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf,
    hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf, int north_xlo, int north_xhi, int north_ylo,
    int north_yhi, hydro2d::DeviceArray2D<double> &north_density0_buf, hydro2d::DeviceArray2D<double> &north_energy0_buf,
    hydro2d::DeviceArray2D<double> &north_pressure_buf, hydro2d::DeviceArray2D<double> &north_viscosity_buf,
    hydro2d::DeviceArray2D<double> &north_soundspeed_buf, hydro2d::DeviceArray2D<double> &north_density1_buf,
    hydro2d::DeviceArray2D<double> &north_energy1_buf, hydro2d::DeviceArray2D<double> &north_velx0_buf, hydro2d::DeviceArray2D<double> &north_vely0_buf,
    hydro2d::DeviceArray2D<double> &north_velx1_buf, hydro2d::DeviceArray2D<double> &north_vely1_buf, hydro2d::DeviceArray2D<double> &north_volume_flux_x_buf,
    hydro2d::DeviceArray2D<double> &north_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &north_mass_flux_x_buf,
    hydro2d::DeviceArray2D<double> &north_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  // Density 0
  if (exchange_mask[fld_density0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *north_density0 = north_density0_buf.data;
    size_t north_density0_nx = north_density0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : local_density0_buf.N()], north_density0[ : north_density0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density0[(x_lo - j) + (k)*local_density0_nx] = north_density0[(north_xhi + 1 - j) + (k)*north_density0_nx];
      }
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *north_density1 = north_density1_buf.data;
    size_t north_density1_nx = north_density1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : local_density1_buf.N()], north_density1[ : north_density1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density1[(x_lo - j) + (k)*local_density1_nx] = north_density1[(north_xhi + 1 - j) + (k)*north_density1_nx];
      }
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *north_energy0 = north_energy0_buf.data;
    size_t north_energy0_nx = north_energy0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : local_energy0_buf.N()], north_energy0[ : north_energy0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        energy0[(x_lo - j) + (k)*local_energy0_nx] = north_energy0[(north_xhi + 1 - j) + (k)*north_energy0_nx];
      }
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *north_energy1 = north_energy1_buf.data;
    size_t north_energy1_nx = north_energy1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(spec_energy1[ : local_energy1_buf.N()], north_energy1[ : north_energy1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        spec_energy1[(x_lo - j) + (k)*local_energy1_nx] = north_energy1[(north_xhi + 1 - j) + (k)*north_energy1_nx];
      }
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *north_pressure = north_pressure_buf.data;
    size_t north_pressure_nx = north_pressure_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(cell_pressure[ : local_pressure_buf.N()], north_pressure[ : north_pressure_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        cell_pressure[(x_lo - j) + (k)*local_pressure_nx] = north_pressure[(north_xhi + 1 - j) + (k)*north_pressure_nx];
      }
    }
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *north_viscosity = north_viscosity_buf.data;
    size_t north_viscosity_nx = north_viscosity_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(shock_damping[ : local_viscosity_buf.N()], north_viscosity[ : north_viscosity_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        shock_damping[(x_lo - j) + (k)*local_viscosity_nx] = north_viscosity[(north_xhi + 1 - j) + (k)*north_viscosity_nx];
      }
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *north_sound_speed = north_soundspeed_buf.data;
    size_t north_soundspeed_nx = north_soundspeed_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(wave_speed[ : local_soundspeed_buf.N()], north_sound_speed[ : north_soundspeed_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        wave_speed[(x_lo - j) + (k)*local_soundspeed_nx] = north_sound_speed[(north_xhi + 1 - j) + (k)*north_soundspeed_nx];
      }
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *north_velx0 = north_velx0_buf.data;
    size_t north_velx0_nx = north_velx0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx0[ : local_velx0_buf.N()], north_velx0[ : north_velx0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx0[(x_lo - j) + (k)*local_velx0_nx] = north_velx0[(north_xhi + 1 - j) + (k)*north_velx0_nx];
      }
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *north_velx1 = north_velx1_buf.data;
    size_t north_velx1_nx = north_velx1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx1[ : local_velx1_buf.N()], north_velx1[ : north_velx1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx1[(x_lo - j) + (k)*local_velx1_nx] = north_velx1[(north_xhi + 1 - j) + (k)*north_velx1_nx];
      }
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *north_vely0 = north_vely0_buf.data;
    size_t north_vely0_nx = north_vely0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely0[ : local_vely0_buf.N()], north_vely0[ : north_vely0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely0[(x_lo - j) + (k)*local_vely0_nx] = north_vely0[(north_xhi + 1 - j) + (k)*north_vely0_nx];
      }
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *north_vely1 = north_vely1_buf.data;
    size_t north_vely1_nx = north_vely1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely1[ : local_vely1_buf.N()], north_vely1[ : north_vely1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely1[(x_lo - j) + (k)*local_vely1_nx] = north_vely1[(north_xhi + 1 - j) + (k)*north_vely1_nx];
      }
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *north_volume_flux_x = north_volume_flux_x_buf.data;
    size_t north_volume_flux_x_nx = north_volume_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_x[ : local_volume_flux_x_buf.N()], north_volume_flux_x[ : north_volume_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_x[(x_lo - j) + (k)*local_volume_flux_x_nx] = north_volume_flux_x[(north_xhi + 1 - j) + (k)*north_volume_flux_x_nx];
      }
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *north_mass_flux_x = north_mass_flux_x_buf.data;
    size_t north_mass_flux_x_nx = north_mass_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_x[ : local_mass_flux_x_buf.N()], north_mass_flux_x[ : north_mass_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_x[(x_lo - j) + (k)*local_mass_flux_x_nx] = north_mass_flux_x[(north_xhi + 1 - j) + (k)*north_mass_flux_x_nx];
      }
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *north_volume_flux_y = north_volume_flux_y_buf.data;
    size_t north_volume_flux_y_nx = north_volume_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_y[ : local_volume_flux_y_buf.N()], north_volume_flux_y[ : north_volume_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_y[(x_lo - j) + (k)*local_volume_flux_y_nx] = north_volume_flux_y[(north_xhi + 1 - j) + (k)*north_volume_flux_y_nx];
      }
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *north_mass_flux_y = north_mass_flux_y_buf.data;
    size_t north_mass_flux_y_nx = north_mass_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_y[ : local_mass_flux_y_buf.N()], north_mass_flux_y[ : north_mass_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_y[(x_lo - j) + (k)*local_mass_flux_y_nx] = north_mass_flux_y[(north_xhi + 1 - j) + (k)*north_mass_flux_y_nx];
      }
    }
  }
}

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
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *south_density0 = south_density0_buf.data;
    size_t south_density0_nx = south_density0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density0[ : local_density0_buf.N()], south_density0[ : south_density0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density0[(x_lo - j) + (k)*local_density0_nx] = south_density0[(south_xhi + 1 - j) + (k)*south_density0_nx];
      }
    }
  }

  // Density 1
  if (exchange_mask[fld_density1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *south_density1 = south_density1_buf.data;
    size_t south_density1_nx = south_density1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(density1[ : local_density1_buf.N()], south_density1[ : south_density1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        density1[(x_lo - j) + (k)*local_density1_nx] = south_density1[(south_xhi + 1 - j) + (k)*south_density1_nx];
      }
    }
  }

  // Energy 0
  if (exchange_mask[fld_energy0] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *south_energy0 = south_energy0_buf.data;
    size_t south_energy0_nx = south_energy0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(energy0[ : local_energy0_buf.N()], south_energy0[ : south_energy0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        energy0[(x_lo - j) + (k)*local_energy0_nx] = south_energy0[(south_xhi + 1 - j) + (k)*south_energy0_nx];
      }
    }
  }

  // Energy 1
  if (exchange_mask[fld_energy1] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *south_energy1 = south_energy1_buf.data;
    size_t south_energy1_nx = south_energy1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(spec_energy1[ : local_energy1_buf.N()], south_energy1[ : south_energy1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        spec_energy1[(x_lo - j) + (k)*local_energy1_nx] = south_energy1[(south_xhi + 1 - j) + (k)*south_energy1_nx];
      }
    }
  }

  // Pressure
  if (exchange_mask[fld_pressure] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *south_pressure = south_pressure_buf.data;
    size_t south_pressure_nx = south_pressure_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(cell_pressure[ : local_pressure_buf.N()], south_pressure[ : south_pressure_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        cell_pressure[(x_lo - j) + (k)*local_pressure_nx] = south_pressure[(south_xhi + 1 - j) + (k)*south_pressure_nx];
      }
    }
  }

  // Viscosity
  if (exchange_mask[fld_viscosity] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *south_viscosity = south_viscosity_buf.data;
    size_t south_viscosity_nx = south_viscosity_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(shock_damping[ : local_viscosity_buf.N()], south_viscosity[ : south_viscosity_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        shock_damping[(x_lo - j) + (k)*local_viscosity_nx] = south_viscosity[(south_xhi + 1 - j) + (k)*south_viscosity_nx];
      }
    }
  }

  // Soundspeed
  if (exchange_mask[fld_sound_speed] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *south_sound_speed = south_soundspeed_buf.data;
    size_t south_soundspeed_nx = south_soundspeed_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(wave_speed[ : local_soundspeed_buf.N()], south_sound_speed[ : south_soundspeed_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        wave_speed[(x_lo - j) + (k)*local_soundspeed_nx] = south_sound_speed[(south_xhi + 1 - j) + (k)*south_soundspeed_nx];
      }
    }
  }

  // XVEL 0
  if (exchange_mask[fld_velx0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *south_velx0 = south_velx0_buf.data;
    size_t south_velx0_nx = south_velx0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx0[ : local_velx0_buf.N()], south_velx0[ : south_velx0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx0[(x_lo - j) + (k)*local_velx0_nx] = south_velx0[(south_xhi + 1 - j) + (k)*south_velx0_nx];
      }
    }
  }

  // XVEL 1
  if (exchange_mask[fld_velx1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *south_velx1 = south_velx1_buf.data;
    size_t south_velx1_nx = south_velx1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(velx1[ : local_velx1_buf.N()], south_velx1[ : south_velx1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx1[(x_lo - j) + (k)*local_velx1_nx] = south_velx1[(south_xhi + 1 - j) + (k)*south_velx1_nx];
      }
    }
  }

  // YVEL 0
  if (exchange_mask[fld_vely0] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *south_vely0 = south_vely0_buf.data;
    size_t south_vely0_nx = south_vely0_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely0[ : local_vely0_buf.N()], south_vely0[ : south_vely0_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely0[(x_lo - j) + (k)*local_vely0_nx] = south_vely0[(south_xhi + 1 - j) + (k)*south_vely0_nx];
      }
    }
  }

  // YVEL 1
  if (exchange_mask[fld_vely1] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *south_vely1 = south_vely1_buf.data;
    size_t south_vely1_nx = south_vely1_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(vely1[ : local_vely1_buf.N()], south_vely1[ : south_vely1_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely1[(x_lo - j) + (k)*local_vely1_nx] = south_vely1[(south_xhi + 1 - j) + (k)*south_vely1_nx];
      }
    }
  }

  // VOL_FLUX_X
  if (exchange_mask[fld_volume_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *south_volume_flux_x = south_volume_flux_x_buf.data;
    size_t south_volume_flux_x_nx = south_volume_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_x[ : local_volume_flux_x_buf.N()], south_volume_flux_x[ : south_volume_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_x[(x_lo - j) + (k)*local_volume_flux_x_nx] = south_volume_flux_x[(south_xhi + 1 - j) + (k)*south_volume_flux_x_nx];
      }
    }
  }

  // MASS_FLUX_X
  if (exchange_mask[fld_mass_flux_x] == 1) {
    // DO k=y_lo-halo_width,y_hi+halo_width

    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *south_mass_flux_x = south_mass_flux_x_buf.data;
    size_t south_mass_flux_x_nx = south_mass_flux_x_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_x[ : local_mass_flux_x_buf.N()], south_mass_flux_x[ : south_mass_flux_x_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_x[(x_lo - j) + (k)*local_mass_flux_x_nx] = south_mass_flux_x[(south_xhi + 1 - j) + (k)*south_mass_flux_x_nx];
      }
    }
  }

  // VOL_FLUX_Y
  if (exchange_mask[fld_volume_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *south_volume_flux_y = south_volume_flux_y_buf.data;
    size_t south_volume_flux_y_nx = south_volume_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(volume_flux_y[ : local_volume_flux_y_buf.N()], south_volume_flux_y[ : south_volume_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_y[(x_lo - j) + (k)*local_volume_flux_y_nx] = south_volume_flux_y[(south_xhi + 1 - j) + (k)*south_volume_flux_y_nx];
      }
    }
  }

  // MASS_FLUX_Y
  if (exchange_mask[fld_mass_flux_y] == 1) {
    // DO k=y_lo-halo_width,y_hi+1+halo_width

    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *south_mass_flux_y = south_mass_flux_y_buf.data;
    size_t south_mass_flux_y_nx = south_mass_flux_y_buf.nX();
#pragma acc parallel loop gang worker vector clover_use_target(globals.context.use_target) \
    present(face_mass_flux_y[ : local_mass_flux_y_buf.N()], south_mass_flux_y[ : south_mass_flux_y_buf.N()])
    for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_y[(x_lo - j) + (k)*local_mass_flux_y_nx] = south_mass_flux_y[(south_xhi + 1 - j) + (k)*south_mass_flux_y_nx];
      }
    }
  }
}
