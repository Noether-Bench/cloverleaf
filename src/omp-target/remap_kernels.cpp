
#include "remap_sweeps.h"

#include <cmath>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace {

void compute_sweep_volumes(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, bool use_target) {

  const size_t cell_row_pitch = field.cell_row_pitch;
  const size_t node_row_pitch = field.node_row_pitch;
  const size_t xflux_row_pitch = field.xflux_row_pitch;
  const size_t yface_row_stride = field.yface_row_stride;

  if (dir == glb_dir_x) {
    if (sweep_index == 1) {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_x = field.volume_flux_x.data;
      double *volume_flux_y = field.volume_flux_y.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
          end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        }
      }
    }
  } else if (dir == glb_dir_y) {
    if (sweep_index == 1) {

      double *cell_volume = field.cell_volume.data;
      double *volume_flux_x = field.volume_flux_x.data;
      double *volume_flux_y = field.volume_flux_y.data;
      double *start_volume = field.scratch_buf1.data;
      double *end_volume = field.scratch_buf2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          start_volume[i + j * node_row_pitch] =
              cell_volume[i + j * cell_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
          end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        }
      }
    }
  }
}

void compute_momentum_volumes(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, int momentum_pass_id, field_set &field) {

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;
  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;

  if (momentum_pass_id == 1) {

    double *volume_flux_y = field.volume_flux_y.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] =
            cell_volume[i + j * cell_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
      }
    }
  } else if (momentum_pass_id == 2) {

    double *volume_flux_y = field.volume_flux_y.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] =
            cell_volume[i + j * cell_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
      }
    }
  } else if (momentum_pass_id == 3) {

    double *volume_flux_y = field.volume_flux_y.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_y[(i + 0) + (j + 1) * yface_row_stride] - volume_flux_y[i + j * yface_row_stride];
      }
    }
  } else if (momentum_pass_id == 4) {

    double *volume_flux_x = field.volume_flux_x.data;
    double *cell_volume = field.cell_volume.data;
    double *start_volume = field.scratch_buf5.data;
    double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
      for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
        end_volume[i + j * node_row_pitch] = cell_volume[i + j * cell_row_pitch];
        start_volume[i + j * node_row_pitch] =
            end_volume[i + j * node_row_pitch] + volume_flux_x[(i + 1) + (j + 0) * xflux_row_pitch] - volume_flux_x[i + j * xflux_row_pitch];
      }
    }
  }
}

void remap_cell_fields_kernel(field_set &field, int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, bool use_target) {

  const double sixth = 1.0 / 6.0;

  const size_t cell_row_pitch = field.cell_row_pitch;
  const size_t node_row_pitch = field.node_row_pitch;
  const size_t xflux_row_pitch = field.xflux_row_pitch;
  const size_t yface_row_stride = field.yface_row_stride;

  compute_sweep_volumes(field, x_lo, x_hi, y_lo, y_hi, dir, sweep_index, use_target);

  if (dir == glb_dir_x) {

    double *dxnode = field.dxnode.data;
    double *density1 = field.density1.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *face_mass_flux_x = field.face_mass_flux_x.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *start_volume = field.scratch_buf1.data;
    double *face_energy_flux = field.scratch_buf7.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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


#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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

    double *dynode = field.dynode.data;
    double *density1 = field.density1.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *face_mass_flux_y = field.face_mass_flux_y.data;
    double *volume_flux_y = field.volume_flux_y.data;
    double *start_volume = field.scratch_buf1.data;
    double *face_energy_flux = field.scratch_buf7.data;
#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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


#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
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

void remap_momentum_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &vel_comp_buffer, field_set &field,
                      int vel_component, int sweep_index, int sweep_dir) {

  int momentum_pass_id = sweep_dir + 2 * (sweep_index - 1);


  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;
  const int xflux_row_pitch = field.xflux_row_pitch;
  const int yface_row_stride = field.yface_row_stride;

  const int node_x_begin = x_lo + 1;
  const int node_x_end = x_hi + 1 + 2;
  const int node_y_begin = y_lo + 1;
  const int node_y_end = y_hi + 1 + 2;

  compute_momentum_volumes(use_target, x_lo, x_hi, y_lo, y_hi, momentum_pass_id, field);

  if (sweep_dir == 1) {
    if (vel_component == 1) {

      double *face_mass_flux_x = field.face_mass_flux_x.data;
      double *nodal_mass_flux = field.scratch_buf1.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = node_y_begin; j < node_y_end; j++) {
        for (int i = (x_lo - 2 + 1); i < (x_hi + 2 + 2); i++) {
          nodal_mass_flux[i + j * node_row_pitch] =
              0.25 * (face_mass_flux_x[(i + 0) + (j - 1) * xflux_row_pitch] + face_mass_flux_x[i + j * xflux_row_pitch] +
                      face_mass_flux_x[(i + 1) + (j - 1) * xflux_row_pitch] + face_mass_flux_x[(i + 1) + (j + 0) * xflux_row_pitch]);
        }
      }


      double *density1 = field.density1.data;
      double *nodal_mass_after = field.scratch_buf2.data;
      double *nodal_mass_before = field.scratch_buf3.data;
      double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = node_y_begin; j < node_y_end; j++) {
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


    const size_t vel_comp_nx = vel_comp_buffer.nX();
    double *vel_comp = vel_comp_buffer.data;
    double *nodal_mass_flux = field.scratch_buf1.data;
    double *nodal_mass_before = field.scratch_buf3.data;
    double *momentum_flux = field.scratch_buf4.data;
    double *dxcell = field.dxcell.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = node_y_begin; j < node_y_end; j++) {
      for (int i = (x_lo - 1 + 1); i < node_x_end; i++)
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


    double *nodal_mass_after = field.scratch_buf2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = node_y_begin; j < node_y_end; j++) {
      for (int i = node_x_begin; i < node_x_end; i++) {
        vel_comp[i + j * vel_comp_nx] = (vel_comp[i + j * vel_comp_nx] * nodal_mass_before[i + j * node_row_pitch] +
                                    momentum_flux[(i - 1) + (j + 0) * node_row_pitch] - momentum_flux[i + j * node_row_pitch]) /
                                   nodal_mass_after[i + j * node_row_pitch];
      }
    }
  } else if (sweep_dir == 2) {
    if (vel_component == 1) {

      double *nodal_mass_flux = field.scratch_buf1.data;
      double *face_mass_flux_y = field.face_mass_flux_y.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_lo - 2 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = node_x_begin; i < node_x_end; i++) {
          nodal_mass_flux[i + j * node_row_pitch] =
              0.25 * (face_mass_flux_y[(i - 1) + (j + 0) * yface_row_stride] + face_mass_flux_y[i + j * yface_row_stride] +
                      face_mass_flux_y[(i - 1) + (j + 1) * yface_row_stride] + face_mass_flux_y[(i + 0) + (j + 1) * yface_row_stride]);
        }
      }

      double *density1 = field.density1.data;
      double *nodal_mass_after = field.scratch_buf2.data;
      double *nodal_mass_before = field.scratch_buf3.data;
      double *end_volume = field.scratch_buf6.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
      for (int j = (y_lo - 1 + 1); j < (y_hi + 2 + 2); j++) {
        for (int i = node_x_begin; i < node_x_end; i++) {
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


    const size_t vel_comp_nx = vel_comp_buffer.nX();
    double *vel_comp = vel_comp_buffer.data;
    double *nodal_mass_flux = field.scratch_buf1.data;
    double *nodal_mass_before = field.scratch_buf3.data;
    double *momentum_flux = field.scratch_buf4.data;
    double *dycell = field.dycell.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = (y_lo - 1 + 1); j < node_y_end; j++) {
      for (int i = node_x_begin; i < node_x_end; i++)
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


    double *nodal_mass_after = field.scratch_buf2.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = node_y_begin; j < node_y_end; j++) {
      for (int i = node_x_begin; i < node_x_end; i++) {
        vel_comp[i + j * vel_comp_nx] = (vel_comp[i + j * vel_comp_nx] * nodal_mass_before[i + j * node_row_pitch] +
                                    momentum_flux[(i + 0) + (j - 1) * node_row_pitch] - momentum_flux[i + j * node_row_pitch]) /
                                   nodal_mass_after[i + j * node_row_pitch];
      }
    }
  }
}

} // namespace

void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir) {

#if HOST_DEVICE_SYNC
  globals.pushToDevice();
#endif

  tile_block &t = globals.subdomain.tile_list[tile_id];
  remap_cell_fields_kernel(t.field, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, sweep_dir, sweep_index,
                    globals.context.use_target);

#if HOST_DEVICE_SYNC
  globals.pullFromDevice();
#endif
}

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
