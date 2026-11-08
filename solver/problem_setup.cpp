

#include <algorithm>
#include <sstream>
#include <string>

#include "field_storage.h"
#include "halo_exchange.h"
#include "field_output.h"
#include "problem_setup.h"
#include "cell_state.h"
#include "ghost_cells.h"

extern std::ostream glb_log;

sim_state setup_run(mpi_context &parallel, const run_config &settings, hydro2d::context ctx) {

  if (parallel.is_root) {
    glb_log << "Setting up initial shape_selector" << std::endl << std::endl;
  }

  hydro2d_sync();

  int first_col, last_col, first_row, top;
  auto subdomainAdjacent = hydro2d_partition(settings, parallel, settings.mesh_spec.grid_nx, settings.mesh_spec.grid_ny, first_col, last_col, first_row, top);


  int grid_nx = last_col - first_col + 1;
  int grid_ny = top - first_row + 1;

  sim_state globals(settings, ctx,
                           subdomain_type(subdomainAdjacent, parallel.rank_id, 1, 1, grid_nx, grid_ny, first_col, last_col, first_row, top, 1,
                                      settings.mesh_spec.grid_nx, 1, settings.mesh_spec.grid_ny, settings.tiles_per_subdomain));

  auto tile_layouts = hydro2d_partition_tiles(globals, grid_nx, grid_ny);

  std::transform(tile_layouts.begin(), tile_layouts.end(), std::back_inserter(globals.subdomain.tile_list),
                 [&](const tile_desc &ti) { return tile_block(ti, globals.context); });

  allocate_fields(globals);

  hydro2d_sync(globals);

  hydro2d_alloc_comm_buffers(globals, parallel);

  if (parallel.is_root) {
    glb_log << "Generating chunks" << std::endl;
  }

  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_00_field_storage.txt");
  for (int tile_id = 0; tile_id < settings.tiles_per_subdomain; ++tile_id) {
    setup_subdomain_mesh(tile_id, globals);
    populate_subdomain(tile_id, globals);
  }
  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_01_initial_fields.txt");

  hydro2d_sync(globals);

  bool saved_profiling = globals.timing_enabled;
  globals.timing_enabled = false;

  for (int tile_id = 0; tile_id < settings.tiles_per_subdomain; ++tile_id) {
    equation_of_state(globals, tile_id, false);
  }
  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_02_cell_state.txt");

  int exchange_mask[FIELD_COUNT];
  for (int &field : exchange_mask)
    field = 0;

  exchange_mask[fld_density0] = 1;
  exchange_mask[fld_energy0] = 1;
  exchange_mask[fld_pressure] = 1;
  exchange_mask[fld_viscosity] = 1;
  exchange_mask[fld_density1] = 1;
  exchange_mask[fld_energy1] = 1;
  exchange_mask[fld_velx0] = 1;
  exchange_mask[fld_vely0] = 1;
  exchange_mask[fld_velx1] = 1;
  exchange_mask[fld_vely1] = 1;

  refresh_ghost_cells(globals, exchange_mask, 2);
  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_03_update_halo.txt");

  if (parallel.is_root) {
    std::cout << " Problem initialised and generated" << std::endl;
    glb_log << std::endl << "Problem initialised and generated" << std::endl;
  }

  report_field_totals(globals, parallel);
  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_04_field_summary.txt");

  if (settings.viz_interval != 0) plot_output(globals, parallel);

  hydro2d_sync(globals);

  globals.timing_enabled = saved_profiling;

  return globals;
}
