#include "tile_seams.h"

using field_grid = hydro2d::DeviceArray2D<double>;

void copy_tile_ghosts_south_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, field_grid &local_density0_buf,
    field_grid &local_energy0_buf, field_grid &local_pressure_buf, field_grid &local_viscosity_buf,
    field_grid &local_soundspeed_buf, field_grid &local_density1_buf, field_grid &local_energy1_buf,
    field_grid &local_velx0_buf, field_grid &local_vely0_buf, field_grid &local_velx1_buf,
    field_grid &local_vely1_buf, field_grid &local_volume_flux_x_buf, field_grid &local_volume_flux_y_buf,
    field_grid &local_mass_flux_x_buf, field_grid &local_mass_flux_y_buf, int south_xlo, int south_xhi,
    int south_ylo, int south_yhi, field_grid &south_density0_buf, field_grid &south_energy0_buf,
    field_grid &south_pressure_buf, field_grid &south_viscosity_buf,
    field_grid &south_soundspeed_buf, field_grid &south_density1_buf,
    field_grid &south_energy1_buf, field_grid &south_velx0_buf,
    field_grid &south_vely0_buf, field_grid &south_velx1_buf,
    field_grid &south_vely1_buf, field_grid &south_volume_flux_x_buf,
    field_grid &south_volume_flux_y_buf, field_grid &south_mass_flux_x_buf,
    field_grid &south_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  if (exchange_mask[fld_density0] == 1) {
    double *density0 = local_density0_buf.data;
    size_t local_density0_nx = local_density0_buf.nX();
    double *south_density0 = south_density0_buf.data;
    size_t south_density0_nx = south_density0_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        density0[j + (y_lo - k) * local_density0_nx] = south_density0[j + (south_yhi + 1 - k) * south_density0_nx];
      }
    }
  }

  if (exchange_mask[fld_density1] == 1) {
    double *density1 = local_density1_buf.data;
    size_t local_density1_nx = local_density1_buf.nX();
    double *south_density1 = south_density1_buf.data;
    size_t south_density1_nx = south_density1_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        density1[j + (y_lo - k) * local_density1_nx] = south_density1[j + (south_yhi + 1 - k) * south_density1_nx];
      }
    }
  }

  if (exchange_mask[fld_energy0] == 1) {
    double *energy0 = local_energy0_buf.data;
    size_t local_energy0_nx = local_energy0_buf.nX();
    double *south_energy0 = south_energy0_buf.data;
    size_t south_energy0_nx = south_energy0_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        energy0[j + (y_lo - k) * local_energy0_nx] = south_energy0[j + (south_yhi + 1 - k) * south_energy0_nx];
      }
    }
  }

  if (exchange_mask[fld_energy1] == 1) {
    double *spec_energy1 = local_energy1_buf.data;
    size_t local_energy1_nx = local_energy1_buf.nX();
    double *south_energy1 = south_energy1_buf.data;
    size_t south_energy1_nx = south_energy1_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        spec_energy1[j + (y_lo - k) * local_energy1_nx] = south_energy1[j + (south_yhi + 1 - k) * south_energy1_nx];
      }
    }
  }

  if (exchange_mask[fld_pressure] == 1) {
    double *cell_pressure = local_pressure_buf.data;
    size_t local_pressure_nx = local_pressure_buf.nX();
    double *south_pressure = south_pressure_buf.data;
    size_t south_pressure_nx = south_pressure_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        cell_pressure[j + (y_lo - k) * local_pressure_nx] = south_pressure[j + (south_yhi + 1 - k) * south_pressure_nx];
      }
    }
  }

  if (exchange_mask[fld_viscosity] == 1) {
    double *shock_damping = local_viscosity_buf.data;
    size_t local_viscosity_nx = local_viscosity_buf.nX();
    double *south_viscosity = south_viscosity_buf.data;
    size_t south_viscosity_nx = south_viscosity_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        shock_damping[j + (y_lo - k) * local_viscosity_nx] = south_viscosity[j + (south_yhi + 1 - k) * south_viscosity_nx];
      }
    }
  }

  if (exchange_mask[fld_sound_speed] == 1) {
    double *wave_speed = local_soundspeed_buf.data;
    size_t local_soundspeed_nx = local_soundspeed_buf.nX();
    double *south_sound_speed = south_soundspeed_buf.data;
    size_t south_soundspeed_nx = south_soundspeed_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        wave_speed[j + (y_lo - k) * local_soundspeed_nx] = south_sound_speed[j + (south_yhi + 1 - k) * south_soundspeed_nx];
      }
    }
  }

  if (exchange_mask[fld_velx0] == 1) {
    double *velx0 = local_velx0_buf.data;
    size_t local_velx0_nx = local_velx0_buf.nX();
    double *south_velx0 = south_velx0_buf.data;
    size_t south_velx0_nx = south_velx0_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        velx0[j + (y_lo - k) * local_velx0_nx] = south_velx0[j + (south_yhi + 1 - k) * south_velx0_nx];
      }
    }
  }

  if (exchange_mask[fld_velx1] == 1) {
    double *velx1 = local_velx1_buf.data;
    size_t local_velx1_nx = local_velx1_buf.nX();
    double *south_velx1 = south_velx1_buf.data;
    size_t south_velx1_nx = south_velx1_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        velx1[j + (y_lo - k) * local_velx1_nx] = south_velx1[j + (south_yhi + 1 - k) * south_velx1_nx];
      }
    }
  }

  if (exchange_mask[fld_vely0] == 1) {
    double *vely0 = local_vely0_buf.data;
    size_t local_vely0_nx = local_vely0_buf.nX();
    double *south_vely0 = south_vely0_buf.data;
    size_t south_vely0_nx = south_vely0_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        vely0[j + (y_lo - k) * local_vely0_nx] = south_vely0[j + (south_yhi + 1 - k) * south_vely0_nx];
      }
    }
  }

  if (exchange_mask[fld_vely1] == 1) {
    double *vely1 = local_vely1_buf.data;
    size_t local_vely1_nx = local_vely1_buf.nX();
    double *south_vely1 = south_vely1_buf.data;
    size_t south_vely1_nx = south_vely1_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        vely1[j + (y_lo - k) * local_vely1_nx] = south_vely1[j + (south_yhi + 1 - k) * south_vely1_nx];
      }
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {
    double *volume_flux_x = local_volume_flux_x_buf.data;
    size_t local_volume_flux_x_nx = local_volume_flux_x_buf.nX();
    double *south_volume_flux_x = south_volume_flux_x_buf.data;
    size_t south_volume_flux_x_nx = south_volume_flux_x_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        volume_flux_x[j + (y_lo - k) * local_volume_flux_x_nx] = south_volume_flux_x[j + (south_yhi + 1 - k) * south_volume_flux_x_nx];
      }
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {
    double *face_mass_flux_x = local_mass_flux_x_buf.data;
    size_t local_mass_flux_x_nx = local_mass_flux_x_buf.nX();
    double *south_mass_flux_x = south_mass_flux_x_buf.data;
    size_t south_mass_flux_x_nx = south_mass_flux_x_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        face_mass_flux_x[j + (y_lo - k) * local_mass_flux_x_nx] = south_mass_flux_x[j + (south_yhi + 1 - k) * south_mass_flux_x_nx];
      }
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {
    double *volume_flux_y = local_volume_flux_y_buf.data;
    size_t local_volume_flux_y_nx = local_volume_flux_y_buf.nX();
    double *south_volume_flux_y = south_volume_flux_y_buf.data;
    size_t south_volume_flux_y_nx = south_volume_flux_y_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        volume_flux_y[j + (y_lo - k) * local_volume_flux_y_nx] = south_volume_flux_y[j + (south_yhi + 1 - k) * south_volume_flux_y_nx];
      }
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {
    double *face_mass_flux_y = local_mass_flux_y_buf.data;
    size_t local_mass_flux_y_nx = local_mass_flux_y_buf.nX();
    double *south_mass_flux_y = south_mass_flux_y_buf.data;
    size_t south_mass_flux_y_nx = south_mass_flux_y_buf.nX();
    for (int k = 0; k < halo_width; ++k) {
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        face_mass_flux_y[j + (y_lo - k) * local_mass_flux_y_nx] = south_mass_flux_y[j + (south_yhi + 1 - k) * south_mass_flux_y_nx];
      }
    }
  }
}

static void fill_rows_from_north(sim_state &globals, int x_lo, int x_hi, int y_hi, int north_ylo, int x_stagger, int y_stagger,
                                 field_grid &ghost_buf, field_grid &north_buf, int halo_width) {
  for (int k = 0; k < halo_width; ++k) {

    double *ghost = ghost_buf.data;
    size_t ghost_nx = ghost_buf.nX();
    double *north = north_buf.data;
    size_t north_nx = north_buf.nX();
#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + x_stagger + halo_width + 2); j++) {
      ghost[j + (y_hi + y_stagger + 2 + k) * ghost_nx] = north[j + (north_ylo + y_stagger - 1 + 2 + k) * north_nx];
    }
  }
}

void copy_tile_ghosts_north_kernel(
    sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, field_grid &local_density0_buf,
    field_grid &local_energy0_buf, field_grid &local_pressure_buf, field_grid &local_viscosity_buf,
    field_grid &local_soundspeed_buf, field_grid &local_density1_buf, field_grid &local_energy1_buf,
    field_grid &local_velx0_buf, field_grid &local_vely0_buf, field_grid &local_velx1_buf,
    field_grid &local_vely1_buf, field_grid &local_volume_flux_x_buf, field_grid &local_volume_flux_y_buf,
    field_grid &local_mass_flux_x_buf, field_grid &local_mass_flux_y_buf, int north_xlo, int north_xhi, int north_ylo,
    int north_yhi, field_grid &north_density0_buf, field_grid &north_energy0_buf,
    field_grid &north_pressure_buf, field_grid &north_viscosity_buf,
    field_grid &north_soundspeed_buf, field_grid &north_density1_buf,
    field_grid &north_energy1_buf, field_grid &north_velx0_buf, field_grid &north_vely0_buf,
    field_grid &north_velx1_buf, field_grid &north_vely1_buf, field_grid &north_volume_flux_x_buf,
    field_grid &north_volume_flux_y_buf, field_grid &north_mass_flux_x_buf,
    field_grid &north_mass_flux_y_buf, const int exchange_mask[FIELD_COUNT], int halo_width) {
  if (exchange_mask[fld_density0] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_density0_buf, north_density0_buf, halo_width);

  if (exchange_mask[fld_density1] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_density1_buf, north_density1_buf, halo_width);

  if (exchange_mask[fld_energy0] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_energy0_buf, north_energy0_buf, halo_width);

  if (exchange_mask[fld_energy1] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_energy1_buf, north_energy1_buf, halo_width);

  if (exchange_mask[fld_pressure] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_pressure_buf, north_pressure_buf, halo_width);

  if (exchange_mask[fld_viscosity] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_viscosity_buf, north_viscosity_buf, halo_width);

  if (exchange_mask[fld_sound_speed] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 0, local_soundspeed_buf, north_soundspeed_buf, halo_width);

  if (exchange_mask[fld_velx0] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 1, local_velx0_buf, north_velx0_buf, halo_width);

  if (exchange_mask[fld_velx1] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 1, local_velx1_buf, north_velx1_buf, halo_width);

  if (exchange_mask[fld_vely0] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 1, local_vely0_buf, north_vely0_buf, halo_width);

  if (exchange_mask[fld_vely1] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 1, local_vely1_buf, north_vely1_buf, halo_width);

  if (exchange_mask[fld_volume_flux_x] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 0, local_volume_flux_x_buf, north_volume_flux_x_buf, halo_width);

  if (exchange_mask[fld_mass_flux_x] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 1, 0, local_mass_flux_x_buf, north_mass_flux_x_buf, halo_width);

  if (exchange_mask[fld_volume_flux_y] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 1, local_volume_flux_y_buf, north_volume_flux_y_buf, halo_width);

  if (exchange_mask[fld_mass_flux_y] == 1)
    fill_rows_from_north(globals, x_lo, x_hi, y_hi, north_ylo, 0, 1, local_mass_flux_y_buf, north_mass_flux_y_buf, halo_width);
}
