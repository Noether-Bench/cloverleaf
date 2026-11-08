#include "tile_seams.h"

using seam_array = hydro2d::DeviceArray2D<double>;

static void copy_column_from_west(sim_state &globals, int x_lo, int y_lo, int row_end, int west_xhi, seam_array &ghost_buf,
                                  seam_array &west_buf, int halo_width) {
  double *ghost = ghost_buf.data;
  size_t ghost_nx = ghost_buf.nX();
  double *west = west_buf.data;
  size_t west_nx = west_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int k = (y_lo - halo_width + 1); k < row_end; k++) {
    for (int j = 0; j < halo_width; ++j) {
      ghost[(x_lo - j) + (k)*ghost_nx] = west[(west_xhi + 1 - j) + (k)*west_nx];
    }
  }
}

void copy_tile_ghosts_west_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               seam_array &local_density0_buf, seam_array &local_energy0_buf,
                               seam_array &local_pressure_buf, seam_array &local_viscosity_buf,
                               seam_array &local_soundspeed_buf, seam_array &local_density1_buf,
                               seam_array &local_energy1_buf, seam_array &local_velx0_buf,
                               seam_array &local_vely0_buf, seam_array &local_velx1_buf,
                               seam_array &local_vely1_buf, seam_array &local_volume_flux_x_buf,
                               seam_array &local_volume_flux_y_buf, seam_array &local_mass_flux_x_buf,
                               seam_array &local_mass_flux_y_buf, int west_xlo, int west_xhi, int west_ylo, int west_yhi,
                               seam_array &west_density0_buf, seam_array &west_energy0_buf,
                               seam_array &west_pressure_buf, seam_array &west_viscosity_buf,
                               seam_array &west_soundspeed_buf, seam_array &west_density1_buf,
                               seam_array &west_energy1_buf, seam_array &west_velx0_buf,
                               seam_array &west_vely0_buf, seam_array &west_velx1_buf,
                               seam_array &west_vely1_buf, seam_array &west_volume_flux_x_buf,
                               seam_array &west_volume_flux_y_buf, seam_array &west_mass_flux_x_buf,
                               seam_array &west_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  if (exchange_mask[fld_density0] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_density0_buf, west_density0_buf, halo_width);

  if (exchange_mask[fld_density1] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_density1_buf, west_density1_buf, halo_width);

  if (exchange_mask[fld_energy0] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_energy0_buf, west_energy0_buf, halo_width);

  if (exchange_mask[fld_energy1] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_energy1_buf, west_energy1_buf, halo_width);

  if (exchange_mask[fld_pressure] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_pressure_buf, west_pressure_buf, halo_width);

  if (exchange_mask[fld_viscosity] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_viscosity_buf, west_viscosity_buf, halo_width);

  if (exchange_mask[fld_sound_speed] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_soundspeed_buf, west_soundspeed_buf, halo_width);

  if (exchange_mask[fld_velx0] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_velx0_buf, west_velx0_buf, halo_width);

  if (exchange_mask[fld_velx1] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_velx1_buf, west_velx1_buf, halo_width);

  if (exchange_mask[fld_vely0] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_vely0_buf, west_vely0_buf, halo_width);

  if (exchange_mask[fld_vely1] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_vely1_buf, west_vely1_buf, halo_width);

  if (exchange_mask[fld_volume_flux_x] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_volume_flux_x_buf, west_volume_flux_x_buf, halo_width);

  if (exchange_mask[fld_mass_flux_x] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + halo_width + 2, west_xhi, local_mass_flux_x_buf, west_mass_flux_x_buf, halo_width);

  if (exchange_mask[fld_volume_flux_y] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_volume_flux_y_buf, west_volume_flux_y_buf, halo_width);

  if (exchange_mask[fld_mass_flux_y] == 1)
    copy_column_from_west(globals, x_lo, y_lo, y_hi + 1 + halo_width + 2, west_xhi, local_mass_flux_y_buf, west_mass_flux_y_buf, halo_width);
}

void copy_tile_ghosts_east_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi,
                               seam_array &local_density0_buf, seam_array &local_energy0_buf,
                               seam_array &local_pressure_buf, seam_array &local_viscosity_buf,
                               seam_array &local_soundspeed_buf, seam_array &local_density1_buf,
                               seam_array &local_energy1_buf, seam_array &local_velx0_buf,
                               seam_array &local_vely0_buf, seam_array &local_velx1_buf,
                               seam_array &local_vely1_buf, seam_array &local_volume_flux_x_buf,
                               seam_array &local_volume_flux_y_buf, seam_array &local_mass_flux_x_buf,
                               seam_array &local_mass_flux_y_buf, int east_xlo, int east_xhi, int east_ylo, int east_yhi,
                               seam_array &east_density0_buf, seam_array &east_energy0_buf,
                               seam_array &east_pressure_buf, seam_array &east_viscosity_buf,
                               seam_array &east_soundspeed_buf, seam_array &east_density1_buf,
                               seam_array &east_energy1_buf, seam_array &east_velx0_buf,
                               seam_array &east_vely0_buf, seam_array &east_velx1_buf,
                               seam_array &east_vely1_buf, seam_array &east_volume_flux_x_buf,
                               seam_array &east_volume_flux_y_buf, seam_array &east_mass_flux_x_buf,
                               seam_array &east_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  const int row_begin = y_lo - halo_width + 1;
  const int cell_row_end = y_hi + halo_width + 2;
  const int node_row_end = y_hi + 1 + halo_width + 2;

  if (exchange_mask[fld_density0] == 1) {

    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *east_density0 = east_density0_buf.data;
    size_t east_density0_nx = east_density0_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        density0[(x_hi + 2 + j) + (k)*local_density0_nx] = east_density0[(east_xlo - 1 + 2 + j) + (k)*east_density0_nx];
      }
    }
  }

  if (exchange_mask[fld_density1] == 1) {

    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *east_density1 = east_density1_buf.data;
    size_t east_density1_nx = east_density1_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        density1[(x_hi + 2 + j) + (k)*local_density1_nx] = east_density1[(east_xlo - 1 + 2 + j) + (k)*east_density1_nx];
      }
    }
  }

  if (exchange_mask[fld_energy0] == 1) {

    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *east_energy0 = east_energy0_buf.data;
    size_t east_energy0_nx = east_energy0_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        energy0[(x_hi + 2 + j) + (k)*local_energy0_nx] = east_energy0[(east_xlo - 1 + 2 + j) + (k)*east_energy0_nx];
      }
    }
  }

  if (exchange_mask[fld_energy1] == 1) {

    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *east_energy1 = east_energy1_buf.data;
    size_t east_energy1_nx = east_energy1_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        spec_energy1[(x_hi + 2 + j) + (k)*local_energy1_nx] = east_energy1[(east_xlo - 1 + 2 + j) + (k)*east_energy1_nx];
      }
    }
  }

  if (exchange_mask[fld_pressure] == 1) {

    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *east_pressure = east_pressure_buf.data;
    size_t east_pressure_nx = east_pressure_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        cell_pressure[(x_hi + 2 + j) + (k)*local_pressure_nx] = east_pressure[(east_xlo - 1 + 2 + j) + (k)*east_pressure_nx];
      }
    }
  }

  if (exchange_mask[fld_viscosity] == 1) {

    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *east_viscosity = east_viscosity_buf.data;
    size_t east_viscosity_nx = east_viscosity_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        shock_damping[(x_hi + 2 + j) + (k)*local_viscosity_nx] = east_viscosity[(east_xlo - 1 + 2 + j) + (k)*east_viscosity_nx];
      }
    }
  }

  if (exchange_mask[fld_sound_speed] == 1) {

    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *east_sound_speed = east_soundspeed_buf.data;
    size_t east_soundspeed_nx = east_soundspeed_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        wave_speed[(x_hi + 2 + j) + (k)*local_soundspeed_nx] = east_sound_speed[(east_xlo - 1 + 2 + j) + (k)*east_soundspeed_nx];
      }
    }
  }

  if (exchange_mask[fld_velx0] == 1) {

    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *east_velx0 = east_velx0_buf.data;
    size_t east_velx0_nx = east_velx0_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx0[(x_hi + 1 + 2 + j) + (k)*local_velx0_nx] = east_velx0[(east_xlo + 1 - 1 + 2 + j) + (k)*east_velx0_nx];
      }
    }
  }

  if (exchange_mask[fld_velx1] == 1) {

    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *east_velx1 = east_velx1_buf.data;
    size_t east_velx1_nx = east_velx1_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        velx1[(x_hi + 1 + 2 + j) + (k)*local_velx1_nx] = east_velx1[(east_xlo + 1 - 1 + 2 + j) + (k)*east_velx1_nx];
      }
    }
  }

  if (exchange_mask[fld_vely0] == 1) {

    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *east_vely0 = east_vely0_buf.data;
    size_t east_vely0_nx = east_vely0_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely0[(x_hi + 1 + 2 + j) + (k)*local_vely0_nx] = east_vely0[(east_xlo + 1 - 1 + 2 + j) + (k)*east_vely0_nx];
      }
    }
  }

  if (exchange_mask[fld_vely1] == 1) {

    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *east_vely1 = east_vely1_buf.data;
    size_t east_vely1_nx = east_vely1_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        vely1[(x_hi + 1 + 2 + j) + (k)*local_vely1_nx] = east_vely1[(east_xlo + 1 - 1 + 2 + j) + (k)*east_vely1_nx];
      }
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {

    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *east_volume_flux_x = east_volume_flux_x_buf.data;
    size_t east_volume_flux_x_nx = east_volume_flux_x_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_x[(x_hi + 1 + 2 + j) + (k)*local_volume_flux_x_nx] =
            east_volume_flux_x[(east_xlo + 1 - 1 + 2 + j) + (k)*east_volume_flux_x_nx];
      }
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {

    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *east_mass_flux_x = east_mass_flux_x_buf.data;
    size_t east_mass_flux_x_nx = east_mass_flux_x_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < cell_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_x[(x_hi + 1 + 2 + j) + (k)*local_mass_flux_x_nx] =
            east_mass_flux_x[(east_xlo + 1 - 1 + 2 + j) + (k)*east_mass_flux_x_nx];
      }
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {

    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *east_volume_flux_y = east_volume_flux_y_buf.data;
    size_t east_volume_flux_y_nx = east_volume_flux_y_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        volume_flux_y[(x_hi + 2 + j) + (k)*local_volume_flux_y_nx] = east_volume_flux_y[(east_xlo - 1 + 2 + j) + (k)*east_volume_flux_y_nx];
      }
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {

    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *east_mass_flux_y = east_mass_flux_y_buf.data;
    size_t east_mass_flux_y_nx = east_mass_flux_y_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int k = row_begin; k < node_row_end; k++) {
      for (int j = 0; j < halo_width; ++j) {
        face_mass_flux_y[(x_hi + 2 + j) + (k)*local_mass_flux_y_nx] = east_mass_flux_y[(east_xlo - 1 + 2 + j) + (k)*east_mass_flux_y_nx];
      }
    }
  }
}
