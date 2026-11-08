#pragma once

#include "sim_state.h"

void equation_of_state(sim_state &globals, const int tile_id, bool predictor_pass);

void shock_damping(sim_state &globals);
