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

void update_halo_kernel_2(sycl::exec_ctx &exec_ctx, int x_lo, int x_hi, int y_lo, int y_hi, const std::array<int, 4> &subdomain_adjacent,
                          const std::array<int, 4> &tile_adjacent, field_set &field, const int exchange_mask[FIELD_COUNT], int halo_width) {

  if (exchange_mask[fld_velx0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.velx0(j, 1 - k) = field.velx0(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.velx0(j, y_hi + 1 + 2 + k) = field.velx0(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.velx0(1 - j, k) = -field.velx0(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.velx0(x_hi + 2 + 1 + j, k) = -field.velx0(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_velx1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.velx1(j, 1 - k) = field.velx1(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.velx1(j, y_hi + 1 + 2 + k) = field.velx1(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.velx1(1 - j, k) = -field.velx1(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.velx1(x_hi + 2 + 1 + j, k) = -field.velx1(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_vely0] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.vely0(j, 1 - k) = -field.vely0(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.vely0(j, y_hi + 1 + 2 + k) = -field.vely0(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.vely0(1 - j, k) = field.vely0(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.vely0(x_hi + 2 + 1 + j, k) = field.vely0(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_vely1] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.vely1(j, 1 - k) = -field.vely1(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.vely1(j, y_hi + 1 + 2 + k) = -field.vely1(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.vely1(1 - j, k) = field.vely1(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.vely1(x_hi + 2 + 1 + j, k) = field.vely1(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_volume_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.volume_flux_x(j, 1 - k) = field.volume_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.volume_flux_x(j, y_hi + 2 + k) = field.volume_flux_x(j, y_hi - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.volume_flux_x(1 - j, k) = -field.volume_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.volume_flux_x(x_hi + j + 1 + 2, k) = -field.volume_flux_x(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_mass_flux_x] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.face_mass_flux_x(j, 1 - k) = field.face_mass_flux_x(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + 1 + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.face_mass_flux_x(j, y_hi + 2 + k) = field.face_mass_flux_x(j, y_hi - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.face_mass_flux_x(1 - j, k) = -field.face_mass_flux_x(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.face_mass_flux_x(x_hi + j + 1 + 2, k) = -field.face_mass_flux_x(x_hi + 1 - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_volume_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.volume_flux_y(j, 1 - k) = -field.volume_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.volume_flux_y(j, y_hi + k + 1 + 2) = -field.volume_flux_y(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.volume_flux_y(1 - j, k) = field.volume_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.volume_flux_y(x_hi + 2 + j, k) = field.volume_flux_y(x_hi - j, k);
        }
      });
    }
  }

  if (exchange_mask[fld_mass_flux_y] == 1) {
    if ((subdomain_adjacent[subdomain_ylo] == mesh_edge) && (tile_adjacent[tile_ylo] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.face_mass_flux_y(j, 1 - k) = -field.face_mass_flux_y(j, 1 + 2 + k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_yhi] == mesh_edge) && (tile_adjacent[tile_yhi] == tile_edge)) {
      // DO j=x_lo-halo_width,x_hi+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{x_lo - halo_width + 1, x_hi + halo_width + 2}, [=](int j) {
        for (int k = 0; k < halo_width; ++k) {
          field.face_mass_flux_y(j, y_hi + k + 1 + 2) = -field.face_mass_flux_y(j, y_hi + 1 - k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xlo] == mesh_edge) && (tile_adjacent[west_tile] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.face_mass_flux_y(1 - j, k) = field.face_mass_flux_y(1 + 2 + j, k);
        }
      });
    }
    if ((subdomain_adjacent[subdomain_xhi] == mesh_edge) && (tile_adjacent[tile_xhi] == tile_edge)) {
      // DO k=y_lo-halo_width,y_hi+1+halo_width

      hydro2d::par_ranged1(exec_ctx, Span1d{y_lo - halo_width + 1, y_hi + 1 + halo_width + 2}, [=](int k) {
        for (int j = 0; j < halo_width; ++j) {
          field.face_mass_flux_y(x_hi + 2 + j, k) = field.face_mass_flux_y(x_hi - j, k);
        }
      });
    }
  }
}
