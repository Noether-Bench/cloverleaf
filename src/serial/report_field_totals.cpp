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

#include "report_field_totals.h"
#include "context.h"
#include "equation_of_state.h"
#include "diagnostics.h"
#include "wall_seconds.h"

#include <cmath>
#include <iomanip>

extern std::ostream glb_log;

//  @brief Fortran field summary_time kernel
//  @author Wayne Gaudin
//  @details The global_sum mass, internal initial_energy, kinetic initial_energy and cell_volume weighted
//  cell_pressure for the subdomain is calculated.
//  @brief Driver for the field summary_time kernels
//  @author Wayne Gaudin
//  @details The user specified field summary_time kernel is invoked here. A summation
//  across all mesh chunks is then performed and the information outputed.
//  If the run is a test problem, the final result is compared with the expected
//  result and the difference output.
//  Note the reference solution is the local_val returned from an Intel compiler with
//  ieee options set on a single core crun.

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

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &t = globals.subdomain.tile_list[tile_id];

    int y_upper = t.meta.blk_ymax;
    int y_lower = t.meta.blk_ymin;
    int x_upper = t.meta.blk_xmax;
    int x_lower = t.meta.tile_xlo;
    field_set &field = t.field;

    /* kernel region */
    for (int idx = (0); idx < ((y_upper - y_lower + 1) * (x_upper - x_lower + 1)); idx++) {
      const int j = x_lower + 1 + idx % (x_upper - x_lower + 1);
      const int k = y_lower + 1 + idx / (x_upper - x_lower + 1);
      double speed_sq = 0.0;
      for (int kv = k; kv <= k + 1; ++kv) {
        for (int jv = j; jv <= j + 1; ++jv) {
          speed_sq += 0.25 * (field.velx0(jv, kv) * field.velx0(jv, kv) + field.vely0(jv, kv) * field.vely0(jv, kv));
        }
      }
      double zone_volume = field.cell_volume(j, k);
      double zone_mass = zone_volume * field.density0(j, k);
      vol += zone_volume;
      mass += zone_mass;
      ie += zone_mass * field.energy0(j, k);
      ke += zone_mass * 0.5 * speed_sq;
      press += zone_volume * field.cell_pressure(j, k);
    }
  }

  hydro2d_reduce_sum(vol);
  hydro2d_reduce_sum(mass);
  hydro2d_reduce_sum(ie);
  hydro2d_reduce_sum(ke);
  hydro2d_reduce_sum(press);

  if (globals.timing_enabled) globals.kernel_timings.summary_time += wall_seconds() - phase_start;

  hydro2d_print_cycle(globals, parallel, vol, mass, ie, ke, press);
}
