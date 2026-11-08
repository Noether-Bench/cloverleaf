#include "halo_exchange.h"

void hydro2d_pack_xlo(sim_state &globals, hydro2d::DeviceArray1D<double> &left_snd_buffer, int tile_id, const int exchange_mask[FIELD_COUNT],
                      int halo_width, int horiz_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.blk_bottom - globals.subdomain.first_row) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                             horiz_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, left_snd_buffer, cell_centred,
                             node_centred, xface_centred, y_face_centred, halo_width, node_centred, horiz_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, left_snd_buffer, cell_centred,
                             node_centred, xface_centred, y_face_centred, halo_width, node_centred, horiz_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, left_snd_buffer, cell_centred,
                             node_centred, xface_centred, y_face_centred, halo_width, node_centred, horiz_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, left_snd_buffer, cell_centred,
                             node_centred, xface_centred, y_face_centred, halo_width, node_centred, horiz_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                             horiz_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                             horiz_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                             horiz_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_pack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, left_snd_buffer,
                             cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                             horiz_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_pack_xhi(sim_state &globals, hydro2d::DeviceArray1D<double> &right_snd_buffer, int tile_id, const int exchange_mask[FIELD_COUNT],
                       int halo_width, int horiz_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.blk_bottom - globals.subdomain.first_row) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              horiz_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                              horiz_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                              horiz_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                              horiz_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                              horiz_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                              horiz_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                              horiz_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                              horiz_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_pack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, right_snd_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                              horiz_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_pack_ylo(sim_state &globals, hydro2d::DeviceArray1D<double> &bottom_snd_buffer, int tile_id, const int exchange_mask[FIELD_COUNT],
                        int halo_width, int vert_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.tile_west_idx - globals.subdomain.first_col) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               vert_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               vert_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               vert_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               vert_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               vert_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                               vert_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                               vert_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                               vert_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_pack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, bottom_snd_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                               vert_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_pack_yhi(sim_state &globals, hydro2d::DeviceArray1D<double> &top_snd_buffer, int tile_id, const int exchange_mask[FIELD_COUNT], int halo_width,
                     int vert_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.tile_west_idx - globals.subdomain.first_col) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                            vert_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                            vert_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, cell_centred, vert_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, cell_centred, vert_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                            vert_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                            vert_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                            vert_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, top_snd_buffer, cell_centred,
                            node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                            vert_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                            vert_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                            vert_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_pack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, top_snd_buffer,
                            cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                            vert_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_unpack_xlo(sim_state &globals, hydro2d::DeviceArray1D<double> &left_rcv_buffer, const int exchange_mask[FIELD_COUNT], int tile_id,
                        int halo_width, int horiz_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.blk_bottom - globals.subdomain.first_row) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                               horiz_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               horiz_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               horiz_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               horiz_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                               horiz_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                               horiz_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                               horiz_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                               horiz_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_unpack_edge_left(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, left_rcv_buffer,
                               cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                               horiz_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_unpack_xhi(sim_state &globals, hydro2d::DeviceArray1D<double> &right_rcv_buffer, const int exchange_mask[FIELD_COUNT], int tile_id,
                         int halo_width, int horiz_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.blk_bottom - globals.subdomain.first_row) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                horiz_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                horiz_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                horiz_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                horiz_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                horiz_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                                horiz_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                                horiz_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                                horiz_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_unpack_edge_right(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, right_rcv_buffer,
                                cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                                horiz_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_unpack_ylo(sim_state &globals, hydro2d::DeviceArray1D<double> &bottom_rcv_buffer, const int exchange_mask[FIELD_COUNT], int tile_id,
                          int halo_width, int vert_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.tile_west_idx - globals.subdomain.first_col) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                                 vert_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                 vert_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                 vert_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                 vert_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred,
                                 vert_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                                 vert_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, bottom_rcv_buffer,
                                 cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                                 vert_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x,
                                 bottom_rcv_buffer, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                                 vert_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_unpack_edge_bottom(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y,
                                 bottom_rcv_buffer, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                                 vert_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}

void hydro2d_unpack_yhi(sim_state &globals, hydro2d::DeviceArray1D<double> &top_rcv_buffer, const int exchange_mask[FIELD_COUNT], int tile_id,
                       int halo_width, int vert_halo_offset[FIELD_COUNT]) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  int tile_shift = (t.meta.tile_west_idx - globals.subdomain.first_col) * halo_width;

  if (exchange_mask[fld_density0] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density0, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_density0] + tile_shift);
  }
  if (exchange_mask[fld_density1] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.density1, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_density1] + tile_shift);
  }
  if (exchange_mask[fld_energy0] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.energy0, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_energy0] + tile_shift);
  }
  if (exchange_mask[fld_energy1] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.spec_energy1, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_energy1] + tile_shift);
  }
  if (exchange_mask[fld_pressure] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.cell_pressure, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_pressure] + tile_shift);
  }
  if (exchange_mask[fld_viscosity] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.shock_damping, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_viscosity] + tile_shift);
  }
  if (exchange_mask[fld_sound_speed] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.wave_speed, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred,
                              vert_halo_offset[fld_sound_speed] + tile_shift);
  }
  if (exchange_mask[fld_velx0] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx0, top_rcv_buffer, cell_centred,
                              node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_velx0] + tile_shift);
  }
  if (exchange_mask[fld_velx1] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, top_rcv_buffer, cell_centred,
                              node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_velx1] + tile_shift);
  }
  if (exchange_mask[fld_vely0] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely0, top_rcv_buffer, cell_centred,
                              node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_vely0] + tile_shift);
  }
  if (exchange_mask[fld_vely1] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, top_rcv_buffer, cell_centred,
                              node_centred, xface_centred, y_face_centred, halo_width, node_centred, vert_halo_offset[fld_vely1] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_x] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_x, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                              vert_halo_offset[fld_volume_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_volume_flux_y] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.volume_flux_y, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                              vert_halo_offset[fld_volume_flux_y] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_x] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_x, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred,
                              vert_halo_offset[fld_mass_flux_x] + tile_shift);
  }
  if (exchange_mask[fld_mass_flux_y] == 1) {
    hydro2d_unpack_edge_top(globals, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.face_mass_flux_y, top_rcv_buffer,
                              cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred,
                              vert_halo_offset[fld_mass_flux_y] + tile_shift);
  }
}
