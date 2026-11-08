
#include "lagrangian.h"
#include "walltime.h"

void advance_velocity_kernel(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, double dt, bool use_target) {

  double half_step = 0.5 * dt;



  const int xface_area_nx = field.xflux_row_pitch;
  const int yface_area_nx = field.yface_row_stride;
  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;

  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;
  double *cell_volume = field.cell_volume.data;
  double *density0 = field.density0.data;
  double *cell_pressure = field.cell_pressure.data;
  double *shock_damping = field.shock_damping.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;
  double *velx1 = field.velx1.data;
  double *vely1 = field.vely1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      double dt_over_mass_s = half_step / ((density0[(i - 1) + (j - 1) * cell_row_pitch] * cell_volume[(i - 1) + (j - 1) * cell_row_pitch] +
                                       density0[(i - 1) + (j + 0) * cell_row_pitch] * cell_volume[(i - 1) + (j + 0) * cell_row_pitch] +
                                       density0[i + j * cell_row_pitch] * cell_volume[i + j * cell_row_pitch] +
                                       density0[(i + 0) + (j - 1) * cell_row_pitch] * cell_volume[(i + 0) + (j - 1) * cell_row_pitch]) *
                                      0.25);
      velx1[i + j * node_row_pitch] =
          velx0[i + j * node_row_pitch] -
          dt_over_mass_s * (xface_area[i + j * xface_area_nx] * (cell_pressure[i + j * cell_row_pitch] - cell_pressure[(i - 1) + (j + 0) * cell_row_pitch]) +
                          xface_area[(i + 0) + (j - 1) * xface_area_nx] *
                              (cell_pressure[(i + 0) + (j - 1) * cell_row_pitch] - cell_pressure[(i - 1) + (j - 1) * cell_row_pitch]));
      vely1[i + j * node_row_pitch] =
          vely0[i + j * node_row_pitch] -
          dt_over_mass_s * (yface_area[i + j * yface_area_nx] * (cell_pressure[i + j * cell_row_pitch] - cell_pressure[(i + 0) + (j - 1) * cell_row_pitch]) +
                          yface_area[(i - 1) + (j + 0) * yface_area_nx] *
                              (cell_pressure[(i - 1) + (j + 0) * cell_row_pitch] - cell_pressure[(i - 1) + (j - 1) * cell_row_pitch]));
      velx1[i + j * node_row_pitch] =
          velx1[i + j * node_row_pitch] -
          dt_over_mass_s * (xface_area[i + j * xface_area_nx] * (shock_damping[i + j * cell_row_pitch] - shock_damping[(i - 1) + (j + 0) * cell_row_pitch]) +
                          xface_area[(i + 0) + (j - 1) * xface_area_nx] *
                              (shock_damping[(i + 0) + (j - 1) * cell_row_pitch] - shock_damping[(i - 1) + (j - 1) * cell_row_pitch]));
      vely1[i + j * node_row_pitch] =
          vely1[i + j * node_row_pitch] -
          dt_over_mass_s * (yface_area[i + j * yface_area_nx] * (shock_damping[i + j * cell_row_pitch] - shock_damping[(i + 0) + (j - 1) * cell_row_pitch]) +
                          yface_area[(i - 1) + (j + 0) * yface_area_nx] *
                              (shock_damping[(i - 1) + (j + 0) * cell_row_pitch] - shock_damping[(i - 1) + (j - 1) * cell_row_pitch]));
    }
  }
}

void advance_velocity(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];

    advance_velocity_kernel(t.field, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, globals.context.use_target);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.accel_time += wall_seconds() - phase_start;
}

void compute_face_fluxes_kernel(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, double dt, bool use_target) {


  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;
  const int node_row_pitch = field.node_row_pitch;

  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;
  double *velx1 = field.velx1.data;
  double *vely1 = field.vely1.data;
  double *volume_flux_x = field.volume_flux_x.data;
  double *volume_flux_y = field.volume_flux_y.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
  for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
      volume_flux_x[i + j * xflux_row_pitch] = 0.25 * dt * xface_area[i + j * xflux_row_pitch] *
                                          (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                           velx1[i + j * node_row_pitch] + velx1[(i + 0) + (j + 1) * node_row_pitch]);
      volume_flux_y[i + j * yface_row_stride] = 0.25 * dt * yface_area[i + j * yface_row_stride] *
                                          (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                           vely1[i + j * node_row_pitch] + vely1[(i + 1) + (j + 0) * node_row_pitch]);
    }
  }
}

void compute_face_fluxes(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    compute_face_fluxes_kernel(t.field, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, globals.context.use_target);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (globals.timing_enabled) globals.kernel_timings.flux_time += wall_seconds() - phase_start;
}
