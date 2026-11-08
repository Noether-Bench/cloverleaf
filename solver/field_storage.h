#pragma once

#include "rank_comm.h"
#include "sim_state.h"

void allocate_fields(sim_state &globals);

void hydro2d_alloc_comm_buffers(sim_state &globals, mpi_context &parallel);

void shutdown_run( sim_state& globals);
