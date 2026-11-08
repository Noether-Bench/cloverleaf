
#include "ghost_cells.h"
#include "tile_seams.h"

void exchange_tile_ghosts(sim_state &globals, int exchange_mask[FIELD_COUNT], int halo_width) {


  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &tt = globals.subdomain.tile_list[tile_id];
    int above_idx = tt.meta.tile_adjacent[tile_yhi];
    int below_idx = tt.meta.tile_adjacent[tile_ylo];

    if (above_idx != tile_edge) {
      tile_block &tup = globals.subdomain.tile_list[above_idx];
      copy_tile_ghosts_north_kernel(
          globals, tt.meta.tile_xlo, tt.meta.blk_xmax, tt.meta.blk_ymin, tt.meta.blk_ymax, tt.field.density0, tt.field.energy0, tt.field.cell_pressure,
          tt.field.shock_damping, tt.field.wave_speed, tt.field.density1, tt.field.spec_energy1, tt.field.velx0, tt.field.vely0, tt.field.velx1,
          tt.field.vely1, tt.field.volume_flux_x, tt.field.volume_flux_y, tt.field.face_mass_flux_x, tt.field.face_mass_flux_y, tup.meta.tile_xlo,
          tup.meta.blk_xmax, tup.meta.blk_ymin, tup.meta.blk_ymax, tup.field.density0, tup.field.energy0, tup.field.cell_pressure, tup.field.shock_damping,
          tup.field.wave_speed, tup.field.density1, tup.field.spec_energy1, tup.field.velx0, tup.field.vely0, tup.field.velx1, tup.field.vely1,
          tup.field.volume_flux_x, tup.field.volume_flux_y, tup.field.face_mass_flux_x, tup.field.face_mass_flux_y, exchange_mask, halo_width);
    }

    if (below_idx != tile_edge) {
      tile_block &below_tile = globals.subdomain.tile_list[below_idx];
      copy_tile_ghosts_south_kernel(globals, tt.meta.tile_xlo, tt.meta.blk_xmax, tt.meta.blk_ymin, tt.meta.blk_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.cell_pressure, tt.field.shock_damping, tt.field.wave_speed, tt.field.density1,
                                tt.field.spec_energy1, tt.field.velx0, tt.field.vely0, tt.field.velx1, tt.field.vely1, tt.field.volume_flux_x,
                                tt.field.volume_flux_y, tt.field.face_mass_flux_x, tt.field.face_mass_flux_y, below_tile.meta.tile_xlo, below_tile.meta.blk_xmax,
                                below_tile.meta.blk_ymin, below_tile.meta.blk_ymax, below_tile.field.density0, below_tile.field.energy0, below_tile.field.cell_pressure,
                                below_tile.field.shock_damping, below_tile.field.wave_speed, below_tile.field.density1, below_tile.field.spec_energy1, below_tile.field.velx0,
                                below_tile.field.vely0, below_tile.field.velx1, below_tile.field.vely1, below_tile.field.volume_flux_x, below_tile.field.volume_flux_y,
                                below_tile.field.face_mass_flux_x, below_tile.field.face_mass_flux_y, exchange_mask, halo_width);
    }
  }


  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    tile_block &tt = globals.subdomain.tile_list[tile_id];
    int tile_west_idx = tt.meta.tile_adjacent[west_tile];
    int right_idx = tt.meta.tile_adjacent[tile_xhi];

    if (tile_west_idx != tile_edge) {
      tile_block &left_tile = globals.subdomain.tile_list[tile_west_idx];
      copy_tile_ghosts_west_kernel(globals, tt.meta.tile_xlo, tt.meta.blk_xmax, tt.meta.blk_ymin, tt.meta.blk_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.cell_pressure, tt.field.shock_damping, tt.field.wave_speed, tt.field.density1,
                                tt.field.spec_energy1, tt.field.velx0, tt.field.vely0, tt.field.velx1, tt.field.vely1, tt.field.volume_flux_x,
                                tt.field.volume_flux_y, tt.field.face_mass_flux_x, tt.field.face_mass_flux_y, left_tile.meta.tile_xlo, left_tile.meta.blk_xmax,
                                left_tile.meta.blk_ymin, left_tile.meta.blk_ymax, left_tile.field.density0, left_tile.field.energy0, left_tile.field.cell_pressure,
                                left_tile.field.shock_damping, left_tile.field.wave_speed, left_tile.field.density1, left_tile.field.spec_energy1, left_tile.field.velx0,
                                left_tile.field.vely0, left_tile.field.velx1, left_tile.field.vely1, left_tile.field.volume_flux_x, left_tile.field.volume_flux_y,
                                left_tile.field.face_mass_flux_x, left_tile.field.face_mass_flux_y, exchange_mask, halo_width);
    }

    if (right_idx != tile_edge) {
      tile_block &right_tile = globals.subdomain.tile_list[right_idx];
      copy_tile_ghosts_east_kernel(globals, tt.meta.tile_xlo, tt.meta.blk_xmax, tt.meta.blk_ymin, tt.meta.blk_ymax, tt.field.density0,
                                tt.field.energy0, tt.field.cell_pressure, tt.field.shock_damping, tt.field.wave_speed, tt.field.density1,
                                tt.field.spec_energy1, tt.field.velx0, tt.field.vely0, tt.field.velx1, tt.field.vely1, tt.field.volume_flux_x,
                                tt.field.volume_flux_y, tt.field.face_mass_flux_x, tt.field.face_mass_flux_y, right_tile.meta.tile_xlo, right_tile.meta.blk_xmax,
                                right_tile.meta.blk_ymin, right_tile.meta.blk_ymax, right_tile.field.density0, right_tile.field.energy0, right_tile.field.cell_pressure,
                                right_tile.field.shock_damping, right_tile.field.wave_speed, right_tile.field.density1, right_tile.field.spec_energy1,
                                right_tile.field.velx0, right_tile.field.vely0, right_tile.field.velx1, right_tile.field.vely1, right_tile.field.volume_flux_x,
                                right_tile.field.volume_flux_y, right_tile.field.face_mass_flux_x, right_tile.field.face_mass_flux_y, exchange_mask, halo_width);
    }
  }
}
