
#include "stable_dt.h"
#include <string>

#include <cmath>


struct cfl_settings {
  double dt_floor;
  double sound;
  double xflow;
  double yflow;
  double divergence;
};

struct cfl_minimum {
  double dt;
  int cause;
  int j;
  int k;
};

cfl_minimum compute_timestep_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, const cfl_settings &cfl, field_set &field) {

  const double dt_floor = cfl.dt_floor;
  const double cfl_sound_factor = cfl.sound;
  const double cfl_ux_factor = cfl.xflow;
  const double cfl_uy_factor = cfl.yflow;
  const double cfl_div_factor = cfl.divergence;

  int too_small = 0;
  double dt_min_val = glb_huge;
  double packed_cell_code = 1.1;


  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;

  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;
  double *dxcell = field.dxcell.data;
  double *dycell = field.dycell.data;
  double *cell_volume = field.cell_volume.data;
  double *density0 = field.density0.data;
  double *shock_damping = field.shock_damping.data;
  double *wave_speed = field.wave_speed.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;

  double dt_min_val0 = dt_min_val;
#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target) map(tofrom : dt_min_val)                   \
    reduction(min : dt_min_val0)
  for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
    for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
      double dsx = dxcell[i];
      double dsy = dycell[j];
      double cc = wave_speed[i + j * cell_row_pitch] * wave_speed[i + j * cell_row_pitch];
      cc = cc + 2.0 * shock_damping[i + j * cell_row_pitch] / density0[i + j * cell_row_pitch];
      cc = fmax(sqrt(cc), glb_tiny);
      double dt_sound = cfl_sound_factor * fmin(dsx, dsy) / cc;
      double div = 0.0;
      double dv1 = (velx0[i + j * node_row_pitch] + velx0[(i + 0) + (j + 1) * node_row_pitch]) * xface_area[i + j * xflux_row_pitch];
      double dv2 = (velx0[(i + 1) + (j + 0) * node_row_pitch] + velx0[(i + 1) + (j + 1) * node_row_pitch]) *
                   xface_area[(i + 1) + (j + 0) * xflux_row_pitch];
      div = div + dv2 - dv1;
      double dt_xflow = cfl_ux_factor * 2.0 * cell_volume[i + j * cell_row_pitch] / fmax(fmax(fabs(dv1), fabs(dv2)), glb_tiny * cell_volume[i + j * cell_row_pitch]);
      dv1 = (vely0[i + j * node_row_pitch] + vely0[(i + 1) + (j + 0) * node_row_pitch]) * yface_area[i + j * yface_row_stride];
      dv2 = (vely0[(i + 0) + (j + 1) * node_row_pitch] + vely0[(i + 1) + (j + 1) * node_row_pitch]) *
            yface_area[(i + 0) + (j + 1) * yface_row_stride];
      div = div + dv2 - dv1;
      double dt_yflow = cfl_uy_factor * 2.0 * cell_volume[i + j * cell_row_pitch] / fmax(fmax(fabs(dv1), fabs(dv2)), glb_tiny * cell_volume[i + j * cell_row_pitch]);
      div = div / (2.0 * cell_volume[i + j * cell_row_pitch]);
      double dt_divergence;
      if (div < -glb_tiny) {
        dt_divergence = cfl_div_factor * (-1.0 / div);
      } else {
        dt_divergence = glb_huge;
      }
      double cell_dt_min = fmin(dt_sound, fmin(dt_xflow, fmin(dt_yflow, fmin(dt_divergence, glb_huge))));
      dt_min_val0 = fmin(cell_dt_min, dt_min_val0);
    }
  }
  dt_min_val = dt_min_val0;

  const int tile_dt_cause = static_cast<int>(10.01 * (packed_cell_code - static_cast<int>(packed_cell_code)));
  packed_cell_code = packed_cell_code - (packed_cell_code - (int)(packed_cell_code));
  const int dtmin_j = ((int)packed_cell_code) % x_hi;
  const int dtmin_k = static_cast<int>(1.f + (packed_cell_code / x_hi));

  if (dt_min_val < dt_floor) too_small = 1;

  if (too_small != 0) {

    auto &xcentre_view = field.xcentre;
    auto &ycentre_view = field.ycentre;
    auto &density0_view = field.density0;
    auto &energy0_view = field.energy0;
    auto &pressure_view = field.cell_pressure;
    auto &soundspeed_view = field.wave_speed;
    auto &velx0_view = field.velx0;
    auto &vely0_view = field.vely0;

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << dtmin_j << " " << dtmin_k << std::endl
              << "x, y                 : " << xcentre_view[dtmin_j] << " " << ycentre_view[dtmin_k] << std::endl
              << "compute_dt : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << velx0_view(dtmin_j, dtmin_k) << " " << vely0_view(dtmin_j, dtmin_k) << std::endl
              << velx0_view(dtmin_j + 1, dtmin_k) << " " << vely0_view(dtmin_j + 1, dtmin_k) << std::endl
              << velx0_view(dtmin_j + 1, dtmin_k + 1) << " " << vely0_view(dtmin_j + 1, dtmin_k + 1) << std::endl
              << velx0_view(dtmin_j, dtmin_k + 1) << " " << vely0_view(dtmin_j, dtmin_k + 1) << std::endl
              << "initial_density, initial_energy, cell_pressure, wave_speed " << std::endl
              << density0_view(dtmin_j, dtmin_k) << " " << energy0_view(dtmin_j, dtmin_k) << " " << pressure_view(dtmin_j, dtmin_k) << " "
              << soundspeed_view(dtmin_j, dtmin_k) << std::endl;
  }

  return {dt_min_val, tile_dt_cause, dtmin_j, dtmin_k};
}

void compute_timestep(sim_state &globals, int tile_id, double &tile_dt, std::string &limiter_name, double &tile_xcoord, double &tile_ycoord, int &dtmin_j,
             int &dtmin_k) {

  tile_dt = glb_huge;

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  tile_block &t = globals.subdomain.tile_list[tile_id];
  const cfl_settings cfl{globals.settings.dt_floor, globals.settings.cfl_sound_factor, globals.settings.cfl_ux_factor, globals.settings.cfl_uy_factor,
                         globals.settings.cfl_div_factor};
  const cfl_minimum tile_limit = compute_timestep_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, cfl, t.field);
  tile_dt = tile_limit.dt;
  dtmin_j = tile_limit.j;
  dtmin_k = tile_limit.k;

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  if (tile_limit.cause == 1) limiter_name = "sound";
  if (tile_limit.cause == 2) limiter_name = "velx";
  if (tile_limit.cause == 3) limiter_name = "vely";
  if (tile_limit.cause == 4) limiter_name = "div";
}
