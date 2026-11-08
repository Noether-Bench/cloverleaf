#pragma once

#include "rank_comm.h"
#include "sim_state.h"

void plot_output(sim_state &globals, mpi_context &parallel);

void report_field_totals(sim_state &globals, mpi_context &parallel);
