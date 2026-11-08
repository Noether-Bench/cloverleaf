#pragma once

#include <string>

#include "rank_comm.h"
#include "sim_state.h"

void compute_dt(sim_state &globals, mpi_context &parallel);

void compute_timestep(sim_state &globals, int tile_id, double &tile_dt, std::string &limiter_name, double &tile_xcoord, double &tile_ycoord, int &dtmin_j,
             int &dtmin_k);
