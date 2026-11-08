
#include "ghost_cells.h"
#include "halo_exchange.h"
#include "walltime.h"

#include <algorithm>

static void reflect_cell_halo(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                              const std::array<int, 4> &tile_adjacent, double *field_data, size_t cell_row_pitch, int halo_width) {
  if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
      for (int k = 0; k < halo_width; ++k) {
        field_data[j + (1 - k) * cell_row_pitch] = field_data[j + (2 + k) * cell_row_pitch];
      }
    }
  }
  if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
    for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
      for (int k = 0; k < halo_width; ++k) {
        field_data[j + (y_hi + 2 + k) * cell_row_pitch] = field_data[j + (y_hi + 1 - k) * cell_row_pitch];
      }
    }
  }
  if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        field_data[(1 - j) + (k)*cell_row_pitch] = field_data[(2 + j) + (k)*cell_row_pitch];
      }
    }
  }
  if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
    for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
      for (int j = 0; j < halo_width; ++j) {
        field_data[(x_hi + 2 + j) + (k)*cell_row_pitch] = field_data[(x_hi + 1 - j) + (k)*cell_row_pitch];
      }
    }
  }
}

void refresh_ghost_cells_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                        const std::array<int, 4> &tile_adjacent, field_set &field, const int exchange_mask[FIELD_COUNT], int halo_width) {

  const size_t cell_row_pitch = field.cell_row_pitch;
  const size_t node_row_pitch = field.node_row_pitch;
  const size_t xflux_row_pitch = field.xflux_row_pitch;
  const size_t yface_row_stride = field.yface_row_stride;

  const bool on_lower_edge = (subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge);
  const bool on_upper_edge = (subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge);
  const bool on_west_edge = (subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge);
  const bool on_east_edge = (subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge);

  if (exchange_mask[fld_density0] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.density0.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_density1] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.density1.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_energy0] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.energy0.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_energy1] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.spec_energy1.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_pressure] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.cell_pressure.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_viscosity] == 1)
    reflect_cell_halo(use_target, x_lo, x_hi, y_lo, y_hi, subdomain_adjacent, tile_adjacent, field.shock_damping.data, cell_row_pitch, halo_width);

  if (exchange_mask[fld_sound_speed] == 1) {
    double *wave_speed = field.wave_speed.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          wave_speed[j + (1 - k) * cell_row_pitch] = wave_speed[j + (+k) * cell_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          wave_speed[j + (y_hi + 2 + k) * cell_row_pitch] = wave_speed[j + (y_hi + 1 - k) * cell_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          wave_speed[(1 - j) + (k)*cell_row_pitch] = wave_speed[(2 + j) + (k)*cell_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          wave_speed[(x_hi + 2 + j) + (k)*cell_row_pitch] = wave_speed[(x_hi + 1 - j) + (k)*cell_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_velx0] == 1) {
    double *velx0 = field.velx0.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx0[j + (1 - k) * node_row_pitch] = velx0[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx0[j + (y_hi + 1 + 2 + k) * node_row_pitch] = velx0[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx0[(1 - j) + (k)*node_row_pitch] = -velx0[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx0[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = -velx0[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_velx1] == 1) {
    double *velx1 = field.velx1.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx1[j + (1 - k) * node_row_pitch] = velx1[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          velx1[j + (y_hi + 1 + 2 + k) * node_row_pitch] = velx1[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx1[(1 - j) + (k)*node_row_pitch] = -velx1[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          velx1[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = -velx1[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_vely0] == 1) {
    double *vely0 = field.vely0.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely0[j + (1 - k) * node_row_pitch] = -vely0[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely0[j + (y_hi + 1 + 2 + k) * node_row_pitch] = -vely0[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely0[(1 - j) + (k)*node_row_pitch] = vely0[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely0[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = vely0[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_vely1] == 1) {
    double *vely1 = field.vely1.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely1[j + (1 - k) * node_row_pitch] = -vely1[j + (1 + 2 + k) * node_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          vely1[j + (y_hi + 1 + 2 + k) * node_row_pitch] = -vely1[j + (y_hi + 1 - k) * node_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely1[(1 - j) + (k)*node_row_pitch] = vely1[(1 + 2 + j) + (k)*node_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          vely1[(x_hi + 2 + 1 + j) + (k)*node_row_pitch] = vely1[(x_hi + 1 - j) + (k)*node_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {
    double *volume_flux_x = field.volume_flux_x.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_x[j + (1 - k) * xflux_row_pitch] = volume_flux_x[j + (1 + 2 + k) * xflux_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_x[j + (y_hi + 2 + k) * xflux_row_pitch] = volume_flux_x[j + (y_hi - k) * xflux_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_x[(1 - j) + (k)*xflux_row_pitch] = -volume_flux_x[(1 + 2 + j) + (k)*xflux_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_x[(x_hi + j + 1 + 2) + (k)*xflux_row_pitch] = -volume_flux_x[(x_hi + 1 - j) + (k)*xflux_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {
    double *face_mass_flux_x = field.face_mass_flux_x.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_x[j + (1 - k) * xflux_row_pitch] = face_mass_flux_x[j + (1 + 2 + k) * xflux_row_pitch];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + 1 + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_x[j + (y_hi + 2 + k) * xflux_row_pitch] = face_mass_flux_x[j + (y_hi - k) * xflux_row_pitch];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_x[(1 - j) + (k)*xflux_row_pitch] = -face_mass_flux_x[(1 + 2 + j) + (k)*xflux_row_pitch];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_x[(x_hi + j + 1 + 2) + (k)*xflux_row_pitch] = -face_mass_flux_x[(x_hi + 1 - j) + (k)*xflux_row_pitch];
        }
      }
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {
    double *volume_flux_y = field.volume_flux_y.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_y[j + (1 - k) * yface_row_stride] = -volume_flux_y[j + (1 + 2 + k) * yface_row_stride];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          volume_flux_y[j + (y_hi + k + 1 + 2) * yface_row_stride] = -volume_flux_y[j + (y_hi + 1 - k) * yface_row_stride];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_y[(1 - j) + (k)*yface_row_stride] = volume_flux_y[(1 + 2 + j) + (k)*yface_row_stride];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          volume_flux_y[(x_hi + 2 + j) + (k)*yface_row_stride] = volume_flux_y[(x_hi - j) + (k)*yface_row_stride];
        }
      }
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {
    double *face_mass_flux_y = field.face_mass_flux_y.data;

    if (on_lower_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_y[j + (1 - k) * yface_row_stride] = -face_mass_flux_y[j + (1 + 2 + k) * yface_row_stride];
        }
      }
    }
    if (on_upper_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int j = (x_lo - halo_width + 1); j < (x_hi + halo_width + 2); j++) {
        for (int k = 0; k < halo_width; ++k) {
          face_mass_flux_y[j + (y_hi + k + 1 + 2) * yface_row_stride] = -face_mass_flux_y[j + (y_hi + 1 - k) * yface_row_stride];
        }
      }
    }
    if (on_west_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_y[(1 - j) + (k)*yface_row_stride] = face_mass_flux_y[(1 + 2 + j) + (k)*yface_row_stride];
        }
      }
    }
    if (on_east_edge) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
      for (int k = (y_lo - halo_width + 1); k < (y_hi + 1 + halo_width + 2); k++) {
        for (int j = 0; j < halo_width; ++j) {
          face_mass_flux_y[(x_hi + 2 + j) + (k)*yface_row_stride] = face_mass_flux_y[(x_hi - j) + (k)*yface_row_stride];
        }
      }
    }
  }
}

void refresh_ghost_cells(sim_state &globals, int exchange_mask[FIELD_COUNT], int halo_width) {

  double phase_start = 0;
  auto close_phase = [&](double &elapsed) {
    if (!globals.timing_enabled) return;
    elapsed += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  };

  if (globals.timing_enabled) phase_start = wall_seconds();
  exchange_tile_ghosts(globals, exchange_mask, halo_width);
  close_phase(globals.kernel_timings.tile_halo_time);

  hydro2d_halo_swap(globals, exchange_mask, halo_width);
  close_phase(globals.kernel_timings.rank_halo_time);

  const auto &neighbours = globals.subdomain.subdomain_adjacent;
  if (std::any_of(neighbours.begin(), neighbours.end(), [](int side) { return side == mesh_edge; })) {

#if HOST_DEVICE_SYNC
    globals.pushToDevice();
#endif

    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      tile_block &t = globals.subdomain.tile_list[tile_id];
      refresh_ghost_cells_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                         globals.subdomain.subdomain_adjacent, t.meta.tile_adjacent, t.field, exchange_mask, halo_width);
    }

#if HOST_DEVICE_SYNC
    globals.pullFromDevice();
#endif
  }

  if (globals.timing_enabled) globals.kernel_timings.local_halo_time += wall_seconds() - phase_start;
}
