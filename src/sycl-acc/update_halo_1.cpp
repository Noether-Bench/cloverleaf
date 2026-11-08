/*
 Crown Copyright 2012 AWE.

 This file is part of Hydro2dMain.

 Hydro2dMain is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either release_info 3 of the License, or (at your option)
 any later release_info.

 Hydro2dMain is distributed in the hope copied it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 Hydro2dMain. If not, see http://www.gnu.org/licenses/.
 */

#include "messaging.h"
#include "messaging_kernel.h"
#include "context.h"
#include "wall_seconds.h"
#include "refresh_ghost_cells.h"
#include "exchange_tile_ghosts.h"

void update_halo_kernel_1(exec_ctx &exec_ctx, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                          const std::array<int, 4> &tile_adjacent, field_set &field, int exchange_mask[FIELD_COUNT], int halo_width) {

  //  Update gathered in external halo cell_total based on halo_width and exchange_mask requested
  //  Even though half of these loops look the wrong way around, it should be noted
  //  copied halo_width is either 1 or 2 so copied it is more efficient to always thread
  //  loop along the mesh edge.
  if (exchange_mask[fld_density0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            density0[j[0]][1 - k] = density0[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            density0[j[0]][y_hi + 2 + k] = density0[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            density0[1 - j][k[0]] = density0[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density0 = field.density0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            density0[x_hi + 2 + j][k[0]] = density0[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_density1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            density1[j[0]][1 - k] = density1[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            density1[j[0]][y_hi + 2 + k] = density1[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            density1[1 - j][k[0]] = density1[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto density1 = field.density1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            density1[x_hi + 2 + j][k[0]] = density1[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_energy0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      //  DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            energy0[j[0]][1 - k] = energy0[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            energy0[j[0]][y_hi + 2 + k] = energy0[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            energy0[1 - j][k[0]] = energy0[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto energy0 = field.energy0.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            energy0[x_hi + 2 + j][k[0]] = energy0[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_energy1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto spec_energy1 = field.spec_energy1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            spec_energy1[j[0]][1 - k] = spec_energy1[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto spec_energy1 = field.spec_energy1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            spec_energy1[j[0]][y_hi + 2 + k] = spec_energy1[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto spec_energy1 = field.spec_energy1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            spec_energy1[1 - j][k[0]] = spec_energy1[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto spec_energy1 = field.spec_energy1.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            spec_energy1[x_hi + 2 + j][k[0]] = spec_energy1[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_pressure] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_pressure = field.cell_pressure.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            cell_pressure[j[0]][1 - k] = cell_pressure[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_pressure = field.cell_pressure.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            cell_pressure[j[0]][y_hi + 2 + k] = cell_pressure[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_pressure = field.cell_pressure.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            cell_pressure[1 - j][k[0]] = cell_pressure[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_pressure = field.cell_pressure.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            cell_pressure[x_hi + 2 + j][k[0]] = cell_pressure[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_viscosity] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto shock_damping = field.shock_damping.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            shock_damping[j[0]][1 - k] = shock_damping[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto shock_damping = field.shock_damping.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            shock_damping[j[0]][y_hi + 2 + k] = shock_damping[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto shock_damping = field.shock_damping.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            shock_damping[1 - j][k[0]] = shock_damping[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto shock_damping = field.shock_damping.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            shock_damping[x_hi + 2 + j][k[0]] = shock_damping[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }

  if (exchange_mask[fld_sound_speed] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto wave_speed = field.wave_speed.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            wave_speed[j[0]][1 - k] = wave_speed[j[0]][2 + k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto wave_speed = field.wave_speed.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](id<1> j) {
          for (int k = 0; k < halo_width; ++k) {
            wave_speed[j[0]][y_hi + 2 + k] = wave_speed[j[0]][y_hi + 1 - k];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto wave_speed = field.wave_speed.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            wave_speed[1 - j][k[0]] = wave_speed[2 + j][k[0]];
          }
        });
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      //  DO k=y_lo-halo_width,y_hi+halo_width
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto wave_speed = field.wave_speed.access<RW>(h);
        hydro2d::par_ranged<class CONCAT_LINE(refresh_ghost_cells)>(h, {y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](id<1> k) {
          for (int j = 0; j < halo_width; ++j) {
            wave_speed[x_hi + 2 + j][k[0]] = wave_speed[x_hi + 1 - j][k[0]];
          }
        });
      });
    }
  }
}
