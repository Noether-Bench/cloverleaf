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

#include "refresh_ghost_cells.h"
#include "messaging.h"
#include "messaging_kernel.h"
#include "context.h"
#include "wall_seconds.h"
#include "exchange_tile_ghosts.h"

void update_halo_kernel_1(sycl::exec_ctx &exec_ctx, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                          const std::array<int, 4> &tile_adjacent, field_set &field, const int exchange_mask[FIELD_COUNT], int halo_width);

void update_halo_kernel_2(sycl::exec_ctx &exec_ctx, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                          const std::array<int, 4> &tile_adjacent, field_set &field, const int exchange_mask[FIELD_COUNT], int halo_width);

//  @brief Driver for the halo updates
//  @author Wayne Gaudin
//  @details Invokes the kernels for the internal and external halo cell_total for
//  the exchange_mask specified.
void refresh_ghost_cells(sim_state &globals, int exchange_mask[FIELD_COUNT], const int halo_width) {

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();
  exchange_tile_ghosts(globals, exchange_mask, halo_width);
  if (globals.timing_enabled) {
    globals.kernel_timings.tile_halo_time += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  }

  hydro2d_halo_swap(globals, exchange_mask, halo_width);

  if (globals.timing_enabled) {
    globals.kernel_timings.rank_halo_time += wall_seconds() - phase_start;
    phase_start = wall_seconds();
  }

  if ((globals.subdomain.subdomain_adjacent[subdomain_xlo] == mesh_edge) || (globals.subdomain.subdomain_adjacent[subdomain_xhi] == mesh_edge) ||
      (globals.subdomain.subdomain_adjacent[subdomain_ylo] == mesh_edge) || (globals.subdomain.subdomain_adjacent[subdomain_yhi] == mesh_edge)) {

    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      tile_block &t = globals.subdomain.tile_list[tile_id];
      update_halo_kernel_1(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                           globals.subdomain.subdomain_adjacent, t.meta.tile_adjacent, t.field, exchange_mask, halo_width);
      update_halo_kernel_2(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                           globals.subdomain.subdomain_adjacent, t.meta.tile_adjacent, t.field, exchange_mask, halo_width);
    }
  }

  if (globals.timing_enabled) globals.kernel_timings.local_halo_time += wall_seconds() - phase_start;
}
