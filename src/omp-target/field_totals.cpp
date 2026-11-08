
#include "field_output.h"
#include "cell_state.h"
#include "run_log.h"
#include "walltime.h"

#include <cmath>
#include <iomanip>


void report_field_totals(sim_state &globals, mpi_context &parallel) {

  hydro2d_print_cycle_banner(globals, parallel);

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    equation_of_state(globals, tile_id, false);
  }

  if (globals.timing_enabled) {
    globals.kernel_timings.equation_of_state += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  }

  double vol = 0.0;
  double mass = 0.0;
  double ie = 0.0;
  double ke = 0.0;
  double press = 0.0;

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];

    int y_upper = t.meta.blk_ymax;
    int y_lower = t.meta.blk_ymin;
    int x_upper = t.meta.blk_xmax;
    int x_lower = t.meta.tile_xlo;
    field_set &field = t.field;

    const size_t cell_row_pitch = field.cell_row_pitch;
    const size_t node_row_pitch = field.node_row_pitch;

    double *cell_volume = field.cell_volume.data;
    double *density0 = field.density0.data;
    double *energy0 = field.energy0.data;
    double *cell_pressure = field.cell_pressure.data;
    double *velx0 = field.velx0.data;
    double *vely0 = field.vely0.data;

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target) map(tofrom : vol) map(tofrom : mass)   \
    map(tofrom : ie) map(tofrom : ke) map(tofrom : press) reduction(+ : vol, mass, ie, ke, press)
    for (int idx = 0; idx < ((y_upper - y_lower + 1) * (x_upper - x_lower + 1)); idx++) {
      const int j = x_lower + 1 + idx % (x_upper - x_lower + 1);
      const int k = y_lower + 1 + idx / (x_upper - x_lower + 1);
      double speed_sq = 0.0;
      for (int kv = k; kv <= k + 1; ++kv) {
        for (int jv = j; jv <= j + 1; ++jv) {
          speed_sq += 0.25 * (velx0[(jv) + (kv)*node_row_pitch] * velx0[(jv) + (kv)*node_row_pitch] +
                           vely0[(jv) + (kv)*node_row_pitch] * vely0[(jv) + (kv)*node_row_pitch]);
        }
      }
      double zone_volume = cell_volume[j + (k)*cell_row_pitch];
      double zone_mass = zone_volume * density0[j + (k)*cell_row_pitch];
      vol += zone_volume;
      mass += zone_mass;
      ie += zone_mass * energy0[j + (k)*cell_row_pitch];
      ke += zone_mass * 0.5 * speed_sq;
      press += zone_volume * cell_pressure[j + (k)*cell_row_pitch];
    }
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif

  hydro2d_reduce_sum(vol);
  hydro2d_reduce_sum(mass);
  hydro2d_reduce_sum(ie);
  hydro2d_reduce_sum(ke);
  hydro2d_reduce_sum(press);

  if (globals.timing_enabled) globals.kernel_timings.summary_time += wall_seconds() - phase_start;

  hydro2d_print_cycle(globals, parallel, vol, mass, ie, ke, mass);
}
