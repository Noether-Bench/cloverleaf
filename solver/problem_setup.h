#pragma once

#include "rank_comm.h"
#include "sim_state.h"

sim_state setup_run(mpi_context &parallel, const run_config &settings, hydro2d::context ctx);

std::array<int, 4> hydro2d_partition(const run_config &globals, mpi_context &parallel, int grid_nx, int grid_ny, int &first_col, int &last_col,
                                    int &first_row, int &top);

std::vector<tile_desc> hydro2d_partition_tiles(sim_state &globals, int subdomain_ncells_x, int subdomain_ncells_y);

void setup_subdomain_mesh(int tile_id, sim_state &globals);

void populate_subdomain(const int tile_id, sim_state &globals);
