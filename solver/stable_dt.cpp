

#include "stable_dt.h"

#include "cell_state.h"
#include "run_log.h"
#include "walltime.h"
#include "ghost_cells.h"

extern std::ostream glb_log;

void compute_dt(sim_state &globals, mpi_context &parallel) {

  globals.dt = glb_huge;
  int too_small = 0;

  int exchange_mask[FIELD_COUNT];

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    equation_of_state(globals, tile_id, false);
  }

  if (globals.timing_enabled) globals.kernel_timings.equation_of_state += wall_seconds() - phase_start;

  for (int i = 0; i < FIELD_COUNT; ++i)
    exchange_mask[i] = 0;
  exchange_mask[fld_pressure] = 1;
  exchange_mask[fld_energy0] = 1;
  exchange_mask[fld_density0] = 1;
  exchange_mask[fld_velx0] = 1;
  exchange_mask[fld_vely0] = 1;
  refresh_ghost_cells(globals, exchange_mask, 1);

  if (globals.timing_enabled) phase_start = wall_seconds();
  shock_damping(globals);
  if (globals.timing_enabled) globals.kernel_timings.shock_damping += wall_seconds() - phase_start;

  for (int i = 0; i < FIELD_COUNT; ++i)
    exchange_mask[i] = 0;
  exchange_mask[fld_viscosity] = 1;
  refresh_ghost_cells(globals, exchange_mask, 1);

  if (globals.timing_enabled) phase_start = wall_seconds();

  int dtmin_j{}, dtmin_k{};
  double tile_min_dt{};
  double limit_x{}, limit_y{}, tile_xcoord{}, tile_ycoord{};
  std::string limit_reason, tile_dt_cause;
  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    compute_timestep(globals, tile_id, tile_min_dt, tile_dt_cause, tile_xcoord, tile_ycoord, dtmin_j, dtmin_k);

    if (tile_min_dt <= globals.dt) {
      globals.dt = tile_min_dt;
      limit_reason = tile_dt_cause;
      limit_x = tile_xcoord;
      limit_y = tile_ycoord;
      globals.jdt = dtmin_j;
      globals.kdt = dtmin_k;
    }
  }

  globals.dt = std::min(std::min(globals.dt, globals.dt_prev * globals.settings.dt_growth), globals.settings.dt_ceiling);

  hydro2d_reduce_min(globals.dt);
  if (globals.timing_enabled) globals.kernel_timings.compute_dt += wall_seconds() - phase_start;

  if (globals.dt < globals.settings.dt_floor) too_small = 1;

  if (parallel.is_root) {
    glb_log << " Step " << globals.cycle_index << " time " << globals.time << " control " << limit_reason << " compute_dt  " << globals.dt << " "
          << globals.jdt << "," << globals.kdt << " x " << limit_x << " y " << limit_y << std::endl;
    std::cout << " Step " << globals.cycle_index << " time " << globals.time << " control " << limit_reason << " compute_dt  " << globals.dt << " "
              << globals.jdt << "," << globals.kdt << " x " << limit_x << " y " << limit_y << std::endl;
  }
  if (too_small == 1) {
    fatal_message((char *)"compute_dt", (char *)"too_small compute_dt");
  }

  globals.dt_prev = globals.dt;
}
