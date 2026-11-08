#pragma once

#include "sim_state.h"

enum sweep_axis { glb_dir_x = 1, glb_dir_y = 2 };

void flow_sweep_driver(sim_state &globals);

void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir);

void remap_momentum_driver(sim_state &globals, int tile_id, int vel_component, int sweep_dir, int sweep_index);
