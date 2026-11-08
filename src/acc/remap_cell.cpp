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

#include "remap_cell.h"
#include <cmath>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

//  @brief Fortran cell flow_sweep_driver kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void remap_cell_fields_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, field_set &field) {

  const double sixth = 1.0 / 6.0;

  const size_t cell_row_pitch = field.cell_row_pitch;
  const size_t node_row_pitch = field.node_row_pitch;
  const size_t xflux_row_pitch = field.xflux_row_pitch;
  const size_t yface_row_stride = field.yface_row_stride;

  if (dir == glb_dir_x) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2

    if (sweep_index == 1) {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_x = field.volume_flux_x.data;
      double *volume_flux_y = field.volume_flux_y.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(cell_volume[ : field.cell_volume.N()], volume_flux_x[ : field.volume_flux_x.N()],            \
            volume_flux_y[ : field.volume_flux_y.N()], start_volume[ : field.scratch_buf1.N()],    \
            end_volume[ : field.scratch_buf2.N()])
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + (volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch] +
                                             volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride]);
          end_volume[i + j * node_row_pitch] =
              start_volume[i + j * node_row_pitch] - (volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch]);
        }
      }

    } else {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_x = field.volume_flux_x.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(cell_volume[ : field.cell_volume.N()], volume_flux_x[ : field.volume_flux_x.N()],            \
            start_volume[ : field.scratch_buf1.N()], end_volume[ : field.scratch_buf2.N()])
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
          end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        }
      }
    }

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi+2
    double *dxnode = field.dxnode.data;
    double *density1 = field.density1.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *face_mass_flux_x = field.face_mass_flux_x.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *start_volume = field.scratch_buf1.data;
    double *face_energy_flux = field.scratch_buf7.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(dxnode[ : field.dxnode.N()], density1[ : field.density1.N()],          \
            spec_energy1[ : field.spec_energy1.N()], face_mass_flux_x[ : field.face_mass_flux_x.N()],      \
            volume_flux_x[ : field.volume_flux_x.N()], start_volume[ : field.scratch_buf1.N()],    \
            face_energy_flux[ : field.scratch_buf7.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2 + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;
          if (volume_flux_x[i + j * xflux_row_pitch] > 0.0) {
            upstream_cell = i - 2;
            source_cell = i - 1;
            downstream_cell = i;
            dif = source_cell;
          } else {
            upstream_cell = MIN(i + 1, x_hi + 2);
            source_cell = i;
            downstream_cell = i - 1;
            dif = upstream_cell;
          }
          cell_courant = fabs(volume_flux_x[i + j * xflux_row_pitch]) / start_volume[source_cell + j * node_row_pitch];
          limiter_coef3 = (1.0 + cell_courant) * (dxnode[i] / dxnode[dif]);
          limiter_coef4 = 2.0 - cell_courant;
          //					node_courant = cell_courant;
          courant_dens = cell_courant;
          grad_up = density1[source_cell + j * cell_row_pitch] - density1[upstream_cell + j * cell_row_pitch];
          grad_down = density1[downstream_cell + j * cell_row_pitch] - density1[source_cell + j * cell_row_pitch];
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_dens) * slope_sign *
                      fmin(fmin(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_mass_flux_x[i + j * xflux_row_pitch] = volume_flux_x[i + j * xflux_row_pitch] * (density1[source_cell + j * cell_row_pitch] + limit_factor);
          courant_mass = fabs(face_mass_flux_x[i + j * xflux_row_pitch]) / (density1[source_cell + j * cell_row_pitch] * start_volume[source_cell + j * node_row_pitch]);
          grad_up = spec_energy1[source_cell + j * cell_row_pitch] - spec_energy1[upstream_cell + j * cell_row_pitch];
          grad_down = spec_energy1[downstream_cell + j * cell_row_pitch] - spec_energy1[source_cell + j * cell_row_pitch];
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_mass) * slope_sign *
                      fmin(fmin(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_energy_flux[i + j * node_row_pitch] = face_mass_flux_x[i + j * xflux_row_pitch] * (spec_energy1[source_cell + j * cell_row_pitch] + limit_factor);
        });
    }

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(dxnode[ : field.dxnode.N()], density1[ : field.density1.N()],          \
            spec_energy1[ : field.spec_energy1.N()], face_mass_flux_x[ : field.face_mass_flux_x.N()],      \
            volume_flux_x[ : field.volume_flux_x.N()], start_volume[ : field.scratch_buf1.N()],    \
            face_energy_flux[ : field.scratch_buf7.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double start_mass_s = density1[i + j * cell_row_pitch] * start_volume[i + j * node_row_pitch];
        double end_mass_s = start_mass_s + face_mass_flux_x[i + j * xflux_row_pitch] - face_mass_flux_x[(i + 1) + (j + 0) * xflux_row_pitch];
        double end_energy_s = (spec_energy1[i + j * cell_row_pitch] * start_mass_s + face_energy_flux[i + j * node_row_pitch] -
                              face_energy_flux[(i + 1) + (j + 0) * node_row_pitch]) /
                             end_mass_s;
        double remap_volume_s =
            start_volume[i + j * node_row_pitch] + volume_flux_x[i + j * xflux_row_pitch] - volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch];
        density1[i + j * cell_row_pitch] = end_mass_s / remap_volume_s;
        spec_energy1[i + j * cell_row_pitch] = end_energy_s;
      }
    }

  } else if (dir == glb_dir_y) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2

    if (sweep_index == 1) {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_x = field.volume_flux_x.data;
      double *volume_flux_y = field.volume_flux_y.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
      present(end_volume[ : field.scratch_buf2.N()], cell_volume[ : field.cell_volume.N()],         \
              volume_flux_y[: field.volume_flux_y.N()], volume_flux_x[ : field.volume_flux_x.N()], \
              start_volume[ : field.scratch_buf2.N()])
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + (volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride] +
                                             volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch]);
          end_volume[i + j * node_row_pitch] =
              start_volume[i + j * node_row_pitch] - (volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride]);
        }
      }

    } else {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_y = field.volume_flux_y.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
      present(end_volume[ : field.scratch_buf2.N()], cell_volume[ : field.cell_volume.N()],         \
              volume_flux_y[: field.volume_flux_y.N()], start_volume[ : field.scratch_buf2.N()])
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
          end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        }
      }
    }

    // DO k=y_lo,y_hi+2
    //   DO j=x_lo,x_hi
    double *dynode = field.dynode.data;
    double *density1 = field.density1.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *face_mass_flux_y = field.face_mass_flux_y.data;
    double *volume_flux_y = field.volume_flux_y.data;
    double *start_volume = field.scratch_buf1.data;
    double *face_energy_flux = field.scratch_buf7.data;
#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(dynode[ : field.dynode.N()], density1[ : field.density1.N()],          \
            spec_energy1[ : field.spec_energy1.N()], face_mass_flux_y[ : field.face_mass_flux_y.N()],      \
            volume_flux_y[ : field.volume_flux_y.N()], start_volume[ : field.scratch_buf1.N()],    \
            face_energy_flux[ : field.scratch_buf7.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++)
        ({
          int upstream_cell, source_cell, downstream_cell, dif;
          double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;
          if (volume_flux_y[i + j * yface_row_stride] > 0.0) {
            upstream_cell = j - 2;
            source_cell = j - 1;
            downstream_cell = j;
            dif = source_cell;
          } else {
            upstream_cell = MIN(j + 1, y_hi + 2);
            source_cell = j;
            downstream_cell = j - 1;
            dif = upstream_cell;
          }
          cell_courant = fabs(volume_flux_y[i + j * yface_row_stride]) / start_volume[i + source_cell * node_row_pitch];
          limiter_coef3 = (1.0 + cell_courant) * (dynode[j] / dynode[dif]);
          limiter_coef4 = 2.0 - cell_courant;
          //					node_courant = cell_courant;
          courant_dens = cell_courant;
          grad_up = density1[i + source_cell * cell_row_pitch] - density1[i + upstream_cell * cell_row_pitch];
          grad_down = density1[i + downstream_cell * cell_row_pitch] - density1[i + source_cell * cell_row_pitch];
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_dens) * slope_sign *
                      fmin(fmin(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_mass_flux_y[i + j * yface_row_stride] = volume_flux_y[i + j * yface_row_stride] * (density1[i + source_cell * cell_row_pitch] + limit_factor);
          courant_mass = fabs(face_mass_flux_y[i + j * yface_row_stride]) / (density1[i + source_cell * cell_row_pitch] * start_volume[i + source_cell * node_row_pitch]);
          grad_up = spec_energy1[i + source_cell * cell_row_pitch] - spec_energy1[i + upstream_cell * cell_row_pitch];
          grad_down = spec_energy1[i + downstream_cell * cell_row_pitch] - spec_energy1[i + source_cell * cell_row_pitch];
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_mass) * slope_sign *
                      fmin(fmin(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_energy_flux[i + j * node_row_pitch] = face_mass_flux_y[i + j * yface_row_stride] * (spec_energy1[i + source_cell * cell_row_pitch] + limit_factor);
        });
    }

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi

#pragma acc parallel loop gang worker vector collapse(2) clover_use_target(use_target) \
    present(dynode[ : field.dynode.N()], density1[ : field.density1.N()],          \
            spec_energy1[ : field.spec_energy1.N()], face_mass_flux_y[ : field.face_mass_flux_y.N()],      \
            volume_flux_y[ : field.volume_flux_y.N()], start_volume[ : field.scratch_buf1.N()],    \
            face_energy_flux[ : field.scratch_buf7.N()])
    for (int j = (y_lo + 1); j < (y_hi + 2); j++) {
      for (int i = (x_lo + 1); i < (x_hi + 2); i++) {
        double start_mass_s = density1[i + j * cell_row_pitch] * start_volume[i + j * node_row_pitch];
        double end_mass_s = start_mass_s + face_mass_flux_y[i + j * yface_row_stride] - face_mass_flux_y[(i + 0) + (j + 1) * yface_row_stride];
        double end_energy_s = (spec_energy1[i + j * cell_row_pitch] * start_mass_s + face_energy_flux[i + j * node_row_pitch] -
                              face_energy_flux[(i + 0) + (j + 1) * node_row_pitch]) /
                             end_mass_s;
        double remap_volume_s =
            start_volume[i + j * node_row_pitch] + volume_flux_y[i + j * yface_row_stride] - volume_flux_y[(i + 0) + (j + 1) * yface_row_stride];
        density1[i + j * cell_row_pitch] = end_mass_s / remap_volume_s;
        spec_energy1[i + j * cell_row_pitch] = end_energy_s;
      }
    }
  }
}

//  @brief Cell centred flow_sweep_driver solver.
//  @author Wayne Gaudin
//  @details Invokes the user chosen_device flow_sweep_driver kernel.
void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir) {

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  tile_block &t = globals.subdomain.tile_list[tile_id];
  remap_cell_fields_kernel(globals.context.use_target, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, sweep_dir, sweep_index,
                    t.field);

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif
}
