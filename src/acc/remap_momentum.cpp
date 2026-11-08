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
#include <cmath>

//  @brief Fortran momentum flow_sweep_driver kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note copied although start_volume is only set and not used in the update, please
//  leave it in the method.
void remap_momentum_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &vel_comp_buffer, field_set &field,
                      int vel_component, int sweep_index, int sweep_dir) {

  int momentum_pass_id = sweep_dir + 2 * (sweep_index - 1);

  // DO k=y_lo-2,y_hi+2
  //   DO j=x_lo-2,x_hi+2

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;
  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;

  if (momentum_pass_id == 1) { // x 1

    double *volume_flux_y = field.volume_flux_y.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(volume_flux_y[ : field.volume_flux_y.N()], volume_flux_x[ : field.volume_flux_x.N()],  \
            cell_volume[ : field.cell_volume.N()], start_volume[ : field.scratch_buf5.N()],            \
            end_volume[ : field.scratch_buf6.N()])
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] =
            cell_volume[i + j * cell_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
      }
    }
  } else if (momentum_pass_id == 2) { // y 1

    double *volume_flux_y = field.volume_flux_y.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(volume_flux_y[ : field.volume_flux_y.N()], volume_flux_x[ : field.volume_flux_x.N()],  \
            cell_volume[ : field.cell_volume.N()], start_volume[ : field.scratch_buf5.N()],            \
            end_volume[ : field.scratch_buf6.N()])
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] =
            cell_volume[i + j * cell_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
      }
    }
  } else if (momentum_pass_id == 3) { // x 2

    double *volume_flux_y = field.volume_flux_y.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(volume_flux_y[ : field.volume_flux_y.N()], cell_volume[ : field.cell_volume.N()],          \
            start_volume[ : field.scratch_buf5.N()], end_volume[ : field.scratch_buf6.N()])
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
      }
    }
  } else if (momentum_pass_id == 4) { // y 2

    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(volume_flux_x[ : field.volume_flux_x.N()], cell_volume[ : field.cell_volume.N()],          \
            start_volume[ : field.scratch_buf5.N()], end_volume[ : field.scratch_buf6.N()])
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
      }
    }
  }

  if (sweep_dir == 1) {
    if (vel_component == 1) {
      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-2,x_hi+2

      double *face_mass_flux_x = field.face_mass_flux_x.data;
      double *nodal_mass_flux = field.scratch_buf1.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(face_mass_flux_x[ : field.face_mass_flux_x.N()], nodal_mass_flux[ : field.scratch_buf1.N()]) 
      for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          nodal_mass_flux[i + j * node_row_pitch] =
              0.25 * (face_mass_flux_x[(i + 0) + (j - 1) * xflux_row_pitch] + face_mass_flux_x[i + j * xflux_row_pitch] +
                      face_mass_flux_x[(i + 1) + (j - 1) * xflux_row_pitch] + face_mass_flux_x[(i + 1) + (j + 0) * xflux_row_pitch]);
        }
      }

      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-1,x_hi+2

      double *density1 = field.density1.data;
      double *nodal_mass_after = field.scratch_buf2.data;
      double *nodal_mass_before = field.scratch_buf3.data;
      double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)   \
    present(density1[ : field.density1.N()], nodal_mass_after[ : field.scratch_buf2.N()],   \
            nodal_mass_before[ : field.scratch_buf3.N()], end_volume[ : field.scratch_buf6.N()], \
            nodal_mass_flux[ : field.scratch_buf1.N()])
      for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
        for (int i = (x_lo - 1 + 1); i < (x_hi + 2 + 2); i++) {
          nodal_mass_after[i + j * node_row_pitch] =
              0.25 * (density1[(i + 0) + (j - 1) * cell_row_pitch] * end_volume[(i + 0) + (j - 1) * node_row_pitch] +
                      density1[i + j * cell_row_pitch] * end_volume[i + j * node_row_pitch] +
                      density1[(i - 1) + (j - 1) * cell_row_pitch] * end_volume[(i - 1) + (j - 1) * node_row_pitch] +
                      density1[(i - 1) + (j + 0) * cell_row_pitch] * end_volume[(i - 1) + (j + 0) * node_row_pitch]);
          nodal_mass_before[i + j * node_row_pitch] =
              nodal_mass_after[i + j * node_row_pitch] - nodal_mass_flux[(i - 1) + (j + 0) * node_row_pitch] + nodal_mass_flux[i + j * node_row_pitch];
        }
      }
    }

    // DO k=y_lo,y_hi+1
    //  DO j=x_lo-1,x_hi+1

    const size_t vel_comp_nx = vel_comp_buffer.nX();
    double *vel_comp = vel_comp_buffer.data;
    double *nodal_mass_flux = field.scratch_buf1.data;
    double *nodal_mass_before = field.scratch_buf3.data;
    double *momentum_flux = field.scratch_buf4.data;
    double *dxcell = field.dxcell.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)   \
    present(vel_comp[ : vel_comp_buffer.N()], nodal_mass_flux[ : field.scratch_buf1.N()],               \
            nodal_mass_before[ : field.scratch_buf3.N()], momentum_flux[ : field.scratch_buf4.N()], \
            dxcell[ : field.dxcell.N()])
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo - 1 + 1); i < (x_hi + 1 + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;
          if (nodal_mass_flux[i + j * node_row_pitch] < 0.0) {
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
          node_courant = fabs(nodal_mass_flux[i + j * node_row_pitch]) / (nodal_mass_before[source_cell + j * node_row_pitch]);
          cell_width = dxcell[i];
          vel_jump_up = vel_comp[source_cell + j * vel_comp_nx] - vel_comp[upstream_cell + j * vel_comp_nx];
          vel_jump_down = vel_comp[downstream_cell + j * vel_comp_nx] - vel_comp[source_cell + j * vel_comp_nx];
          limit_factor = 0.0;
          if (vel_jump_up * vel_jump_down > 0.0) {
            auw = fabs(vel_jump_up);
            adw = fabs(vel_jump_down);
            slope_sign = 1.0;
            if (vel_jump_down <= 0.0) slope_sign = -1.0;
            limit_factor = slope_sign * fmin(fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dxcell[dif]) / 6.0, auw), adw);
          }
          remap_face_vel = vel_comp[source_cell + j * vel_comp_nx] + (1.0 - node_courant) * limit_factor;
          momentum_flux[i + j * node_row_pitch] = remap_face_vel * nodal_mass_flux[i + j * node_row_pitch];
        });
    }

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    double *nodal_mass_after = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)   \
    present(vel_comp[ : vel_comp_buffer.N()], nodal_mass_before[ : field.scratch_buf3.N()],           \
            momentum_flux[ : field.scratch_buf4.N()], dxcell[ : field.dxcell.N()],             \
            nodal_mass_after[ : field.scratch_buf2.N()])
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
        vel_comp[i + j * vel_comp_nx] = (vel_comp[i + j * vel_comp_nx] * nodal_mass_before[i + j * node_row_pitch] +
                                    momentum_flux[(i - 1) + (j + 0) * node_row_pitch] - momentum_flux[i + j * node_row_pitch]) /
                                   nodal_mass_after[i + j * node_row_pitch];
      }
    }
  } else if (sweep_dir == 2) {
    if (vel_component == 1) {
      // DO k=y_lo-2,y_hi+2
      //   DO j=x_lo,x_hi+1

      double *nodal_mass_flux = field.scratch_buf1.data;
      double *face_mass_flux_y = field.face_mass_flux_y.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(nodal_mass_flux[ : field.scratch_buf1.N()], face_mass_flux_y[ : field.face_mass_flux_y.N()])
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
          nodal_mass_flux[i + j * node_row_pitch] =
              0.25 * (face_mass_flux_y[(i - 1) + (j + 0) * yface_row_stride] + face_mass_flux_y[i + j * yface_row_stride] +
                      face_mass_flux_y[(i - 1) + (j + 1) * yface_row_stride] + face_mass_flux_y[(i + 0) + (j + 1) * yface_row_stride]);
        }
      }

      // DO k=y_lo-1,y_hi+2
      //   DO j=x_lo,x_hi+1
      double *density1 = field.density1.data;
      double *nodal_mass_after = field.scratch_buf2.data;
      double *nodal_mass_before = field.scratch_buf3.data;
      double *end_volume = field.scratch_buf6.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)        \
    present(nodal_mass_flux[ : field.scratch_buf1.N()], density1[ : field.density1.N()],             \
            nodal_mass_after[ : field.scratch_buf2.N()], nodal_mass_before[ : field.scratch_buf3.N()],\
            end_volume[ : field.scratch_buf6.N()])
            
      for (int j = (y_lo - 1 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
          nodal_mass_after[i + j * node_row_pitch] =
              0.25 * (density1[(i + 0) + (j - 1) * cell_row_pitch] * end_volume[(i + 0) + (j - 1) * node_row_pitch] +
                      density1[i + j * cell_row_pitch] * end_volume[i + j * node_row_pitch] +
                      density1[(i - 1) + (j - 1) * cell_row_pitch] * end_volume[(i - 1) + (j - 1) * node_row_pitch] +
                      density1[(i - 1) + (j + 0) * cell_row_pitch] * end_volume[(i - 1) + (j + 0) * node_row_pitch]);
          nodal_mass_before[i + j * node_row_pitch] =
              nodal_mass_after[i + j * node_row_pitch] - nodal_mass_flux[(i + 0) + (j - 1) * node_row_pitch] + nodal_mass_flux[i + j * node_row_pitch];
        }
      }
    }

    // DO k=y_lo-1,y_hi+1
    //   DO j=x_lo,x_hi+1

    const size_t vel_comp_nx = vel_comp_buffer.nX();
    double *vel_comp = vel_comp_buffer.data;
    double *nodal_mass_flux = field.scratch_buf1.data;
    double *nodal_mass_before = field.scratch_buf3.data;
    double *momentum_flux = field.scratch_buf4.data;
    double *dycell = field.dycell.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)   \
    present(vel_comp[ : vel_comp_buffer.N()], nodal_mass_before[ : field.scratch_buf3.N()],           \
            momentum_flux[ : field.scratch_buf4.N()], dycell[ : field.dycell.N()],             \
            nodal_mass_flux[ : field.scratch_buf1.N()])
    for (int j = (y_lo - 1 + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;
          if (nodal_mass_flux[i + j * node_row_pitch] < 0.0) {
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
          node_courant = fabs(nodal_mass_flux[i + j * node_row_pitch]) / (nodal_mass_before[i + source_cell * node_row_pitch]);
          cell_width = dycell[j];
          vel_jump_up = vel_comp[i + source_cell * vel_comp_nx] - vel_comp[i + upstream_cell * vel_comp_nx];
          vel_jump_down = vel_comp[i + downstream_cell * vel_comp_nx] - vel_comp[i + source_cell * vel_comp_nx];
          limit_factor = 0.0;
          if (vel_jump_up * vel_jump_down > 0.0) {
            auw = fabs(vel_jump_up);
            adw = fabs(vel_jump_down);
            slope_sign = 1.0;
            if (vel_jump_down <= 0.0) slope_sign = -1.0;
            limit_factor = slope_sign * fmin(fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dycell[dif]) / 6.0, auw), adw);
          }
          remap_face_vel = vel_comp[i + source_cell * vel_comp_nx] + (1.0 - node_courant) * limit_factor;
          momentum_flux[i + j * node_row_pitch] = remap_face_vel * nodal_mass_flux[i + j * node_row_pitch];
        });
    }

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    double *nodal_mass_after = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target)   \
    present(vel_comp[ : vel_comp_buffer.N()], nodal_mass_before[ : field.scratch_buf3.N()],           \
            momentum_flux[ : field.scratch_buf4.N()], nodal_mass_after[ : field.scratch_buf2.N()])
    for (int j = (y_lo + 1); j < (y_hi + 1 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 1 + 2); i++) {
        vel_comp[i + j * vel_comp_nx] = (vel_comp[i + j * vel_comp_nx] * nodal_mass_before[i + j * node_row_pitch] +
                                    momentum_flux[(i + 0) + (j - 1) * node_row_pitch] - momentum_flux[i + j * node_row_pitch]) /
                                   nodal_mass_after[i + j * node_row_pitch];
      }
    }
  }
}

//  @brief Momentum flow_sweep_driver solver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum flow_sweep_driver kernel.
void remap_momentum_driver(sim_state &globals, int tile_id, int vel_component, int sweep_dir, int sweep_index) {

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  tile_block &t = globals.subdomain.tile_list[tile_id];
  if (vel_component == 1) {
    remap_momentum_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.velx1, t.field,
                     vel_component, sweep_index, sweep_dir);
  } else {
    remap_momentum_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, t.field.vely1, t.field,
                     vel_component, sweep_index, sweep_dir);
  }

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif
}
