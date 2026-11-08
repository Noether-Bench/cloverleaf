#pragma once

#include "sim_state.h"

void refresh_ghost_cells(sim_state &globals, int exchange_mask[FIELD_COUNT], int halo_width);

void exchange_tile_ghosts(sim_state &globals, int exchange_mask[FIELD_COUNT], int halo_width);
