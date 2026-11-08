
#pragma once

#include <fstream>

#include "rank_comm.h"
#include "run_config.h"

void parse_deck(std::ifstream &glb_input, mpi_context &parallel, run_config &globals);
