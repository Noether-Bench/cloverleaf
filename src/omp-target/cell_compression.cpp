
#include "lagrangian.h"
#include "rank_comm.h"
#include "cell_state.h"
#include "run_log.h"
#include "walltime.h"
#include "ghost_cells.h"
#include <array>
#include <cmath>

enum class compression_stage { half, full };

void compression_work_kernel(bool use_target, compression_stage stage, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field) {

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;
  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;


  if (stage == compression_stage::half) {

    double *xface_area = field.xface_area.data;

    double *yface_area = field.yface_area.data;
    double *cell_volume = field.cell_volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *velx0 = field.velx0.data;
    double *vely0 = field.vely0.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double west_face_flux = (xface_area[i + j * xflux_row_pitch] * (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                                            velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch])) *
                           0.25 * dt * 0.5;
        double east_face_flux = (xface_area[(i + 1) + (j + 0) * xflux_row_pitch] *
                             (velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch] +
                              velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch])) *
                            0.25 * dt * 0.5;
        double south_face_flux = (yface_area[i + j * yface_row_stride] * (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                                              vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch])) *
                             0.25 * dt * 0.5;
        double north_face_flux = (yface_area[(i + 0) + (j + 1) * yface_row_stride] *
                           (vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch] +
                            vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch])) *
                          0.25 * dt * 0.5;
        double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
        double volume_ratio_s = cell_volume[i + j * cell_row_pitch] / (cell_volume[i + j * cell_row_pitch] + net_face_flux);
        double inv_cell_volume = 1.0 / cell_volume[i + j * cell_row_pitch];
        double internal_energy_delta = (cell_pressure[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch] +
                                shock_damping[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch]) *
                               net_face_flux * inv_cell_volume;
        spec_energy1[i + j * cell_row_pitch] = energy0[i + j * cell_row_pitch] - internal_energy_delta;
        density1[i + j * cell_row_pitch] = density0[i + j * cell_row_pitch] * volume_ratio_s;
      }
    }

  } else {

    double *xface_area = field.xface_area.data;
    double *yface_area = field.yface_area.data;
    double *cell_volume = field.cell_volume.data;
    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *velx0 = field.velx0.data;
    double *velx1 = field.velx1.data;
    double *vely0 = field.vely0.data;
    double *vely1 = field.vely1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double west_face_flux = (xface_area[i + j * xflux_row_pitch] * (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch] +
                                                            velx1[i + j * node_row_pitch] + velx1[(i + 0) + (j + 1) * node_row_pitch])) *
                           0.25 * dt;
        double east_face_flux = (xface_area[(i + 1) + (j + 0) * xflux_row_pitch] *
                             (velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch] +
                              velx1[(i + 1) + (j + 0) * node_row_pitch] + velx1[(i + 1) + (j + 1) * node_row_pitch])) *
                            0.25 * dt;
        double south_face_flux = (yface_area[i + j * yface_row_stride] * (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch] +
                                                              vely1[i + j * node_row_pitch] + vely1[(i + 1) + (j + 0) * node_row_pitch])) *
                             0.25 * dt;
        double north_face_flux = (yface_area[(i + 0) + (j + 1) * yface_row_stride] *
                           (vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch] +
                            vely1[(i + 0) + (j + 1) * node_row_pitch] + vely1[(i + 1) + (j + 1) * node_row_pitch])) *
                          0.25 * dt;
        double net_face_flux = east_face_flux - west_face_flux + north_face_flux - south_face_flux;
        double volume_ratio_s = cell_volume[i + j * cell_row_pitch] / (cell_volume[i + j * cell_row_pitch] + net_face_flux);
        double inv_cell_volume = 1.0 / cell_volume[i + j * cell_row_pitch];
        double internal_energy_delta = (cell_pressure[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch] +
                                shock_damping[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch]) *
                               net_face_flux * inv_cell_volume;
        spec_energy1[i + j * cell_row_pitch] = energy0[i + j * cell_row_pitch] - internal_energy_delta;
        density1[i + j * cell_row_pitch] = density0[i + j * cell_row_pitch] * volume_ratio_s;
      }
    }
  }
}

static void finish_predictor_pass(sim_state &globals) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();
  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    equation_of_state(globals, tile_id, true);
  }

  if (globals.timing_enabled) globals.kernel_timings.equation_of_state += wall_seconds() - phase_start;

  std::array<int, FIELD_COUNT> exchange_mask{};
  exchange_mask[fld_pressure] = 1;
  refresh_ghost_cells(globals, exchange_mask.data(), 1);

  if (globals.timing_enabled) phase_start = wall_seconds();
  rollback_state(globals);
  if (globals.timing_enabled) globals.kernel_timings.rollback_state += wall_seconds() - phase_start;
}

void PdV(sim_state &globals, bool predictor_pass) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  globals.error_status = 0;

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  const compression_stage stage = predictor_pass ? compression_stage::half : compression_stage::full;
  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];
    compression_work_kernel(globals.context.use_target, stage, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.dt, t.field);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  hydro2d_max_error(globals.error_status);
  if (globals.timing_enabled) globals.kernel_timings.PdV += wall_seconds() - phase_start;

  if (globals.error_status == 1) {
    fatal_message((char *)"PdV", (char *)"error in PdV");
  }

  if (predictor_pass) finish_predictor_pass(globals);
}
