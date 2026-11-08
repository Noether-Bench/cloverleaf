
#include "lagrangian.h"
#include "walltime.h"

namespace {

void rollback_state_kernel(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, bool use_target) {

  const int cell_row_pitch = field.cell_row_pitch;
  double *density0 = field.density0.data;
  double *density1 = field.density1.data;
  double *energy0 = field.energy0.data;
  double *spec_energy1 = field.spec_energy1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
      density1[i + j * cell_row_pitch] = density0[i + j * cell_row_pitch];
      spec_energy1[i + j * cell_row_pitch] = energy0[i + j * cell_row_pitch];
    }
  }
}

void roll_time_level_kernel(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, bool use_target) {

  const int cell_row_pitch = field.cell_row_pitch;
  double *density0 = field.density0.data;
  double *density1 = field.density1.data;
  double *energy0 = field.energy0.data;
  double *spec_energy1 = field.spec_energy1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
      density0[i + j * cell_row_pitch] = density1[i + j * cell_row_pitch];
      energy0[i + j * cell_row_pitch] = spec_energy1[i + j * cell_row_pitch];
    }
  }

  const int node_row_pitch = field.node_row_pitch;
  double *velx0 = field.velx0.data;
  double *velx1 = field.velx1.data;
  double *vely0 = field.vely0.data;
  double *vely1 = field.vely1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      velx0[i + j * node_row_pitch] = velx1[i + j * node_row_pitch];
      vely0[i + j * node_row_pitch] = vely1[i + j * node_row_pitch];
    }
  }
}

} // namespace

void rollback_state(sim_state &globals) {

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    rollback_state_kernel(t.field, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.context.use_target);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif
}

void roll_time_level(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    roll_time_level_kernel(t.field, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.context.use_target);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.reset_time += wall_seconds() - phase_start;
}
