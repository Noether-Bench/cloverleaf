#include "rank_comm.h"
#include "problem_setup.h"

#include <cstdlib>

extern std::ostream glb_log;

std::array<int, 4> hydro2d_partition(const run_config &globals, mpi_context &parallel, int grid_nx, int grid_ny, int &first_col, int &last_col,
                                    int &first_row, int &top) {

  std::array<int, 4> subdomain_adjacent{};

  int total_subdomains = globals.total_subdomains;


  double aspect_ratio = (double)grid_nx / (double)grid_ny;

  int subdomain_cols = total_subdomains;
  int subdomain_rows = 1;

  int divisor_found = 0;

  double split_x, split_y;

  for (int c = 1; c <= total_subdomains; ++c) {
    if (total_subdomains % c == 0) {
      split_x = total_subdomains / (double)c;
      split_y = c;
      if (split_x / split_y <= aspect_ratio) {
        subdomain_rows = c;
        subdomain_cols = total_subdomains / c;
        divisor_found = 1;
        break;
      }
    }
  }

  if (divisor_found == 0 || subdomain_rows == total_subdomains) {
    if (aspect_ratio >= 1.0) {
      subdomain_cols = total_subdomains;
      subdomain_rows = 1;
    } else {
      subdomain_cols = 1;
      subdomain_rows = total_subdomains;
    }
  }

  int base_width_x = grid_nx / subdomain_cols;
  int base_width_y = grid_ny / subdomain_rows;
  int rem_x = grid_nx % subdomain_cols;
  int rem_y = grid_ny % subdomain_rows;


  int extra_x_sum = 0;
  int extra_y_sum = 0;
  int cnk = 1;
  for (int cy = 1; cy <= subdomain_rows; ++cy) {
    for (int cx = 1; cx <= subdomain_cols; ++cx) {
      int extra_x = 0;
      int extra_y = 0;
      if (cx <= rem_x) extra_x = 1;
      if (cy <= rem_y) extra_y = 1;

      if (cnk == parallel.rank_id + 1) {
        first_col = (cx - 1) * base_width_x + 1 + extra_x_sum;
        last_col = first_col + base_width_x - 1 + extra_x;
        first_row = (cy - 1) * base_width_y + 1 + extra_y_sum;
        top = first_row + base_width_y - 1 + extra_y;

        subdomain_adjacent[subdomain_xlo] = subdomain_cols * (cy - 1) + cx - 1;
        subdomain_adjacent[subdomain_xhi] = subdomain_cols * (cy - 1) + cx + 1;
        subdomain_adjacent[subdomain_ylo] = subdomain_cols * (cy - 2) + cx;
        subdomain_adjacent[subdomain_yhi] = subdomain_cols * (cy) + cx;

        if (cx == 1) subdomain_adjacent[subdomain_xlo] = mesh_edge;
        if (cx == subdomain_cols) subdomain_adjacent[subdomain_xhi] = mesh_edge;
        if (cy == 1) subdomain_adjacent[subdomain_ylo] = mesh_edge;
        if (cy == subdomain_rows) subdomain_adjacent[subdomain_yhi] = mesh_edge;
      }

      if (cx <= rem_x) extra_x_sum = extra_x_sum + 1;

      cnk = cnk + 1;
    }
    extra_x_sum = 0;
    if (cy <= rem_y) extra_y_sum = extra_y_sum + 1;
  }

  if (parallel.is_root) {
    glb_log << std::endl
          << "Mesh ratio of " << aspect_ratio << std::endl
          << "Decomposing the mesh into " << subdomain_cols << " by " << subdomain_rows << " chunks" << std::endl
          << "Decomposing the subdomain with " << globals.tiles_per_subdomain << " tile_list" << std::endl
          << std::endl;
  }
  return subdomain_adjacent;
}

std::vector<tile_desc> hydro2d_partition_tiles(sim_state &globals, int subdomain_ncells_x, int subdomain_ncells_y) {

  std::vector<tile_desc> tile_list(globals.settings.tiles_per_subdomain);

  int subdomain_aspect = (double)subdomain_ncells_x / (double)subdomain_ncells_y;

  int tiles_across = globals.settings.tiles_per_subdomain;
  int tiles_down = 1;

  int divisor_found = 0;
  for (int t = 1; t <= globals.settings.tiles_per_subdomain; ++t) {
    if (globals.settings.tiles_per_subdomain % t == 0) {
      int split_x = globals.settings.tiles_per_subdomain / (double)t;
      int split_y = t;
      if (split_x / split_y <= subdomain_aspect) {
        tiles_down = t;
        tiles_across = globals.settings.tiles_per_subdomain / t;
        divisor_found = 1;
        break;
      }
    }
  }

  if (divisor_found == 0 || tiles_down == globals.settings.tiles_per_subdomain) {
    if (subdomain_aspect >= 1.0) {
      tiles_across = globals.settings.tiles_per_subdomain;
      tiles_down = 1;
    } else {
      tiles_across = 1;
      tiles_down = globals.settings.tiles_per_subdomain;
    }
  }

  int subdomain_span_x = subdomain_ncells_x / tiles_across;
  int subdomain_span_y = subdomain_ncells_y / tiles_down;
  int subdomain_rem_x = subdomain_ncells_x % tiles_across;
  int subdomain_rem_y = subdomain_ncells_y % tiles_down;

  int extra_x_sum = 0;
  int extra_y_sum = 0;
  int tile_id = 0;
  for (int ty = 1; ty <= tiles_down; ++ty) {
    for (int tx = 1; tx <= tiles_across; ++tx) {
      int extra_x = 0;
      int extra_y = 0;
      if (tx <= subdomain_rem_x) extra_x = 1;
      if (ty <= subdomain_rem_y) extra_y = 1;

      int first_col = globals.subdomain.first_col + (tx - 1) * subdomain_span_x + extra_x_sum;
      int last_col = first_col + subdomain_span_x - 1 + extra_x;
      int first_row = globals.subdomain.first_row + (ty - 1) * subdomain_span_y + extra_y_sum;
      int top = first_row + subdomain_span_y - 1 + extra_y;

      tile_list[tile_id].tile_adjacent[west_tile] = tiles_across * (ty - 1) + tx - 1;
      tile_list[tile_id].tile_adjacent[tile_xhi] = tiles_across * (ty - 1) + tx + 1;
      tile_list[tile_id].tile_adjacent[tile_ylo] = tiles_across * (ty - 2) + tx;
      tile_list[tile_id].tile_adjacent[tile_yhi] = tiles_across * (ty) + tx;

      for (int i = 0; i < 4; ++i) {
        tile_list[tile_id].edge_mask[i] = 0;
      }

      if (tx == 1) {
        tile_list[tile_id].tile_adjacent[west_tile] = tile_edge;
        tile_list[tile_id].edge_mask[west_tile] = 1;
      }
      if (tx == tiles_across) {
        tile_list[tile_id].tile_adjacent[tile_xhi] = tile_edge;
        tile_list[tile_id].edge_mask[tile_xhi] = 1;
      }
      if (ty == 1) {
        tile_list[tile_id].tile_adjacent[tile_ylo] = tile_edge;
        tile_list[tile_id].edge_mask[tile_ylo] = 1;
      }
      if (ty == tiles_down) {
        tile_list[tile_id].tile_adjacent[tile_yhi] = tile_edge;
        tile_list[tile_id].edge_mask[tile_yhi] = 1;
      }

      if (tx <= subdomain_rem_x) extra_x_sum = extra_x_sum + 1;

      tile_list[tile_id].tile_xlo = 1;
      tile_list[tile_id].blk_xmax = last_col - first_col + 1;
      tile_list[tile_id].blk_ymin = 1;
      tile_list[tile_id].blk_ymax = top - first_row + 1;

      tile_list[tile_id].tile_west_idx = first_col;
      tile_list[tile_id].right_idx = last_col;
      tile_list[tile_id].blk_top = top;
      tile_list[tile_id].blk_bottom = first_row;

      tile_id = tile_id + 1;
    }
    extra_x_sum = 0;
    if (ty <= subdomain_rem_y) extra_y_sum = extra_y_sum + 1;
  }
  return tile_list;
}
