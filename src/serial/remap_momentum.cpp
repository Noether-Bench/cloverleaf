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

#include "remap_momentum.h"
#include "context.h"
#include <cmath>

//  @brief Fortran momentum flow_sweep_driver kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note copied although start_volume is only set and not used in the update, please
//  leave it in the method.
void remap_momentum_kernel(int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &vel_comp, hydro2d::DeviceArray2D<double> &face_mass_flux_x,
                      hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, hydro2d::DeviceArray2D<double> &volume_flux_y,
                      hydro2d::DeviceArray2D<double> &cell_volume, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &nodal_mass_flux,
                      hydro2d::DeviceArray2D<double> &nodal_mass_after, hydro2d::DeviceArray2D<double> &nodal_mass_before, hydro2d::DeviceArray2D<double> &momentum_flux,
                      hydro2d::DeviceArray2D<double> &start_volume, hydro2d::DeviceArray2D<double> &end_volume, hydro2d::DeviceArray1D<double> &dxcell,
                      hydro2d::DeviceArray1D<double> &dycell, int vel_component, int sweep_index, int sweep_dir) {

  int momentum_pass_id = sweep_dir + 2 * (sweep_index - 1);

  // DO k=y_lo-2,y_hi+2
  //   DO j=x_lo-2,x_hi+2

  if (momentum_pass_id == 1) { // x 1

    /* kernel region */
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume(i, j) = cell_volume(i, j) + volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j);
        start_volume(i, j) = end_volume(i, j) + volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j);
      }
    }
  } else if (momentum_pass_id == 2) { // y 1

    /* kernel region */
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume(i, j) = cell_volume(i, j) + volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j);
        start_volume(i, j) = end_volume(i, j) + volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j);
      }
    }
  } else if (momentum_pass_id == 3) { // x 2

    /* kernel region */
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume(i, j) = cell_volume(i, j);
        start_volume(i, j) = end_volume(i, j) + volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j);
      }
    }
  } else if (momentum_pass_id == 4) { // y 2

    /* kernel region */
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume(i, j) = cell_volume(i, j);
        start_volume(i, j) = end_volume(i, j) + volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j);
      }
    }
  }

  if (sweep_dir == 1) {
    if (vel_component == 1) {
      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-2,x_hi+2

      /* kernel region */
      for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          nodal_mass_flux(i, j) = 0.25 * (face_mass_flux_x(i + 0, j - 1) + face_mass_flux_x(i, j) + face_mass_flux_x(i + 1, j - 1) + face_mass_flux_x(i + 1, j + 0));
        }
      }

      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-1,x_hi+2

      /* kernel region */
      for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
        for (int i = (x_lo - 1 + 1); i < (x_hi + 2 + 2); i++) {
          nodal_mass_after(i, j) = 0.25 * (density1(i + 0, j - 1) * end_volume(i + 0, j - 1) + density1(i, j) * end_volume(i, j) +
                                         density1(i - 1, j - 1) * end_volume(i - 1, j - 1) + density1(i - 1, j + 0) * end_volume(i - 1, j + 0));
          nodal_mass_before(i, j) = nodal_mass_after(i, j) - nodal_mass_flux(i - 1, j + 0) + nodal_mass_flux(i, j);
        }
      }
    }

    // DO k=y_lo,y_hi+1
    //  DO j=x_lo-1,x_hi+1

    /* kernel region */
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo - 1 + 1); i < (x_hi + 1 + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;
          if (nodal_mass_flux(i, j) < 0.0) {
            upstream_cell = i + 2;
            source_cell = i + 1;
            downstream_cell = i;
            dif = source_cell;
          } else {
            upstream_cell = i - 1;
            source_cell = i;
            downstream_cell = i + 1;
            dif = upstream_cell;
          }
          node_courant = std::fabs(nodal_mass_flux(i, j)) / (nodal_mass_before(source_cell, j));
          cell_width = dxcell[i];
          vel_jump_up = vel_comp(source_cell, j) - vel_comp(upstream_cell, j);
          vel_jump_down = vel_comp(downstream_cell, j) - vel_comp(source_cell, j);
          limit_factor = 0.0;
          if (vel_jump_up * vel_jump_down > 0.0) {
            auw = std::fabs(vel_jump_up);
            adw = std::fabs(vel_jump_down);
            slope_sign = 1.0;
            if (vel_jump_down <= 0.0) slope_sign = -1.0;
            limit_factor =
                slope_sign * std::fmin(std::fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dxcell[dif]) / 6.0, auw), adw);
          }
          remap_face_vel = vel_comp(source_cell, j) + (1.0 - node_courant) * limit_factor;
          momentum_flux(i, j) = remap_face_vel * nodal_mass_flux(i, j);
        });
    }

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    /* kernel region */
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
        vel_comp(i, j) = (vel_comp(i, j) * nodal_mass_before(i, j) + momentum_flux(i - 1, j + 0) - momentum_flux(i, j)) / nodal_mass_after(i, j);
      }
    }
  } else if (sweep_dir == 2) {
    if (vel_component == 1) {
      // DO k=y_lo-2,y_hi+2
      //   DO j=x_lo,x_hi+1

      /* kernel region */
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
          nodal_mass_flux(i, j) = 0.25 * (face_mass_flux_y(i - 1, j + 0) + face_mass_flux_y(i, j) + face_mass_flux_y(i - 1, j + 1) + face_mass_flux_y(i + 0, j + 1));
        }
      }

      // DO k=y_lo-1,y_hi+2
      //   DO j=x_lo,x_hi+1

      /* kernel region */
      for (int j = (y_lo - 1 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
          nodal_mass_after(i, j) = 0.25 * (density1(i + 0, j - 1) * end_volume(i + 0, j - 1) + density1(i, j) * end_volume(i, j) +
                                         density1(i - 1, j - 1) * end_volume(i - 1, j - 1) + density1(i - 1, j + 0) * end_volume(i - 1, j + 0));
          nodal_mass_before(i, j) = nodal_mass_after(i, j) - nodal_mass_flux(i + 0, j - 1) + nodal_mass_flux(i, j);
        }
      }
    }

    // DO k=y_lo-1,y_hi+1
    //   DO j=x_lo,x_hi+1

    /* kernel region */
    for (int j = (y_lo - 1 + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;
          if (nodal_mass_flux(i, j) < 0.0) {
            upstream_cell = j + 2;
            source_cell = j + 1;
            downstream_cell = j;
            dif = source_cell;
          } else {
            upstream_cell = j - 1;
            source_cell = j;
            downstream_cell = j + 1;
            dif = upstream_cell;
          }
          node_courant = std::fabs(nodal_mass_flux(i, j)) / (nodal_mass_before(i, source_cell));
          cell_width = dycell[j];
          vel_jump_up = vel_comp(i, source_cell) - vel_comp(i, upstream_cell);
          vel_jump_down = vel_comp(i, downstream_cell) - vel_comp(i, source_cell);
          limit_factor = 0.0;
          if (vel_jump_up * vel_jump_down > 0.0) {
            auw = std::fabs(vel_jump_up);
            adw = std::fabs(vel_jump_down);
            slope_sign = 1.0;
            if (vel_jump_down <= 0.0) slope_sign = -1.0;
            limit_factor =
                slope_sign * std::fmin(std::fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dycell[dif]) / 6.0, auw), adw);
          }
          remap_face_vel = vel_comp(i, source_cell) + (1.0 - node_courant) * limit_factor;
          momentum_flux(i, j) = remap_face_vel * nodal_mass_flux(i, j);
        });
    }

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    /* kernel region */
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
        vel_comp(i, j) = (vel_comp(i, j) * nodal_mass_before(i, j) + momentum_flux(i + 0, j - 1) - momentum_flux(i, j)) / nodal_mass_after(i, j);
      }
    }
  }
}

//  @brief Momentum flow_sweep_driver solver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum flow_sweep_driver kernel.
void remap_momentum_driver(sim_state &globals, int tile_id, int vel_component, int sweep_dir, int sweep_index) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  if (vel_component == 1) {
    remap_momentum_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, t.field.face_mass_flux_x, t.field.volume_flux_x,
                     t.field.face_mass_flux_y, t.field.volume_flux_y, t.field.cell_volume, t.field.density1, t.field.scratch_buf1, t.field.scratch_buf2,
                     t.field.scratch_buf3, t.field.scratch_buf4, t.field.scratch_buf5, t.field.scratch_buf6, t.field.dxcell, t.field.dycell,
                     vel_component, sweep_index, sweep_dir);
  } else {
    remap_momentum_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, t.field.face_mass_flux_x, t.field.volume_flux_x,
                     t.field.face_mass_flux_y, t.field.volume_flux_y, t.field.cell_volume, t.field.density1, t.field.scratch_buf1, t.field.scratch_buf2,
                     t.field.scratch_buf3, t.field.scratch_buf4, t.field.scratch_buf5, t.field.scratch_buf6, t.field.dxcell, t.field.dycell,
                     vel_component, sweep_index, sweep_dir);
  }
}
