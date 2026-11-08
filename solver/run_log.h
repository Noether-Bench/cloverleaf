
#pragma once
#include "rank_comm.h"
#include "sim_state.h"
#include <cmath>

extern std::ostream glb_log;

void fatal_message(char *origin_site, char *error);

void hydro2d_print_cycle_banner(sim_state &globals, mpi_context &parallel);

void hydro2d_print_cycle(sim_state &globals, mpi_context &parallel,
                        double vol, double mass, double ie, double ke, double press);
