
#include "remap_sweeps.h"
#include "walltime.h"
#include "ghost_cells.h"

void flow_sweep_driver(sim_state &globals) {

  int sweep_index = 1;
  int sweep_dir;
  if (globals.remap_in_x) sweep_dir = glb_dir_x;
  if (!globals.remap_in_x) sweep_dir = glb_dir_y;

  int velx = glb_dir_x;
  int vely = glb_dir_y;

  int exchange_mask[FIELD_COUNT];
  for (int &field : exchange_mask)
    field = 0;
  exchange_mask[fld_energy1] = 1;
  exchange_mask[fld_density1] = 1;
  exchange_mask[fld_volume_flux_x] = 1;
  exchange_mask[fld_volume_flux_y] = 1;
  refresh_ghost_cells(globals, exchange_mask, 2);

  double phase_start = 0;
  if (globals.timing_enabled) phase_start = wall_seconds();
  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    remap_cell_fields_driver(globals, tile_id, sweep_index, sweep_dir);
  }

  if (globals.timing_enabled) globals.kernel_timings.cell_remap_time += wall_seconds() - phase_start;

  for (int &field : exchange_mask)
    field = 0;
  exchange_mask[fld_density1] = 1;
  exchange_mask[fld_energy1] = 1;
  exchange_mask[fld_velx1] = 1;
  exchange_mask[fld_vely1] = 1;
  exchange_mask[fld_mass_flux_x] = 1;
  exchange_mask[fld_mass_flux_y] = 1;
  refresh_ghost_cells(globals, exchange_mask, 2);

  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    remap_momentum_driver(globals, tile_id, velx, sweep_dir, sweep_index);
    remap_momentum_driver(globals, tile_id, vely, sweep_dir, sweep_index);
  }

  if (globals.timing_enabled) globals.kernel_timings.momentum_remap_time += wall_seconds() - phase_start;

  sweep_index = 2;
  if (globals.remap_in_x) sweep_dir = glb_dir_y;
  if (!globals.remap_in_x) sweep_dir = glb_dir_x;

  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    remap_cell_fields_driver(globals, tile_id, sweep_index, sweep_dir);
  }

  if (globals.timing_enabled) globals.kernel_timings.cell_remap_time += wall_seconds() - phase_start;

  for (int &field : exchange_mask)
    field = 0;
  exchange_mask[fld_density1] = 1;
  exchange_mask[fld_energy1] = 1;
  exchange_mask[fld_velx1] = 1;
  exchange_mask[fld_vely1] = 1;
  exchange_mask[fld_mass_flux_x] = 1;
  exchange_mask[fld_mass_flux_y] = 1;
  refresh_ghost_cells(globals, exchange_mask, 2);

  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    remap_momentum_driver(globals, tile_id, velx, sweep_dir, sweep_index);
    remap_momentum_driver(globals, tile_id, vely, sweep_dir, sweep_index);
  }

  if (globals.timing_enabled) globals.kernel_timings.momentum_remap_time += wall_seconds() - phase_start;
}
