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
#include "context.h"
#include <cmath>

//  @brief Fortran cell flow_sweep_driver kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void remap_cell_fields_kernel(int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, hydro2d::DeviceArray1D<double> &dxnode,
                       hydro2d::DeviceArray1D<double> &dynode, hydro2d::DeviceArray2D<double> &cell_volume, hydro2d::DeviceArray2D<double> &density1,
                       hydro2d::DeviceArray2D<double> &spec_energy1, hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_x,
                       hydro2d::DeviceArray2D<double> &face_mass_flux_y, hydro2d::DeviceArray2D<double> &volume_flux_y, hydro2d::DeviceArray2D<double> &start_volume,
                       hydro2d::DeviceArray2D<double> &end_volume, hydro2d::DeviceArray2D<double> &pre_mass, hydro2d::DeviceArray2D<double> &post_mass,
                       hydro2d::DeviceArray2D<double> &advec_vol, hydro2d::DeviceArray2D<double> &post_ener, hydro2d::DeviceArray2D<double> &face_energy_flux) {

  const double sixth = 1.0 / 6.0;

  if (dir == glb_dir_x) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2

    const hydro2d::Span2d policy(x_lo - 2 + 1, y_lo - 2 + 1, x_hi + 2 + 2, y_hi + 2 + 2);

    if (sweep_index == 1) {

      hydro2d::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
        start_volume(i, j) = cell_volume(i, j) + (volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j) + volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j));
        end_volume(i, j) = start_volume(i, j) - (volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j));
      });

    } else {

      hydro2d::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
        start_volume(i, j) = cell_volume(i, j) + volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j);
        end_volume(i, j) = cell_volume(i, j);
      });
    }

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi+2
    hydro2d::par_ranged2(Span2d{x_lo + 1, y_lo + 1, x_hi + 2 + 2, y_hi + 2}, [=] DEVICE_KERNEL(const int x, const int y) {
      int upstream_cell, source_cell, downstream_cell, dif;
      double cell_courant, limiter_coef3, limiter_coef4, courant_dens,  courant_mass, grad_up, grad_down, limit_factor, slope_sign;

      const int j = x;
      const int k = y;

      if (volume_flux_x(x, y) > 0.0) {
        upstream_cell = j - 2;
        source_cell = j - 1;
        downstream_cell = j;
        dif = source_cell;
      } else {
        upstream_cell = min(j + 1, x_hi + 2); // XXX can't do std::min because CUDA
        source_cell = j;
        downstream_cell = j - 1;
        dif = upstream_cell;
      }

      cell_courant = std::fabs(volume_flux_x(x, y)) / start_volume(source_cell, k);
      limiter_coef3 = (1.0 + cell_courant) * (dxnode[j] / dxnode[dif]);
      limiter_coef4 = 2.0 - cell_courant;

      courant_dens = cell_courant;

      grad_up = density1(source_cell, k) - density1(upstream_cell, k);
      grad_down = density1(downstream_cell, k) - density1(source_cell, k);
      slope_sign = 1.0;
      if (grad_down <= 0.0) slope_sign = -1.0;
      if (grad_up * grad_down > 0.0) {
        limit_factor = (1.0 - courant_dens) * slope_sign *
                  std::fmin(std::fmin(std::fabs(grad_up), std::fabs(grad_down)),
                            sixth * (limiter_coef3 * std::fabs(grad_up) + limiter_coef4 * std::fabs(grad_down)));
      } else {
        limit_factor = 0.0;
      }
      face_mass_flux_x(x, y) = volume_flux_x(x, y) * (density1(source_cell, k) + limit_factor);

      courant_mass = std::fabs(face_mass_flux_x(x, y)) / (density1(source_cell, k) * start_volume(source_cell, k));
      grad_up = spec_energy1(source_cell, k) - spec_energy1(upstream_cell, k);
      grad_down = spec_energy1(downstream_cell, k) - spec_energy1(source_cell, k);
      slope_sign = 1.0;
      if (grad_down <= 0.0) slope_sign = -1.0;
      if (grad_up * grad_down > 0.0) {
        limit_factor = (1.0 - courant_mass) * slope_sign *
                  std::fmin(std::fmin(std::fabs(grad_up), std::fabs(grad_down)),
                            sixth * (limiter_coef3 * std::fabs(grad_up) + limiter_coef4 * std::fabs(grad_down)));
      } else {
        limit_factor = 0.0;
      }

      face_energy_flux(x, y) = face_mass_flux_x(x, y) * (spec_energy1(source_cell, k) + limit_factor);
    });

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi

    hydro2d::par_ranged2(Span2d{x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
      double start_mass_s = density1(i, j) * start_volume(i, j);
      double end_mass_s = start_mass_s + face_mass_flux_x(i, j) - face_mass_flux_x(i + 1, j + 0);
      double end_energy_s = (spec_energy1(i, j) * start_mass_s + face_energy_flux(i, j) - face_energy_flux(i + 1, j + 0)) / end_mass_s;
      double remap_volume_s = start_volume(i, j) + volume_flux_x(i, j) - volume_flux_x(i + 1, j + 0);
      density1(i, j) = end_mass_s / remap_volume_s;
      spec_energy1(i, j) = end_energy_s;
    });

  } else if (dir == glb_dir_y) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2
    hydro2d::Span2d policy(x_lo - 2 + 1, y_lo - 2 + 1, x_hi + 2 + 2, y_hi + 2 + 2);

    if (sweep_index == 1) {

      hydro2d::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
        start_volume(i, j) = cell_volume(i, j) + (volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j) + volume_flux_x(i + 1, j + 0) - volume_flux_x(i, j));
        end_volume(i, j) = start_volume(i, j) - (volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j));
      });

    } else {

      hydro2d::par_ranged2(policy, [=] DEVICE_KERNEL(const int i, const int j) {
        start_volume(i, j) = cell_volume(i, j) + volume_flux_y(i + 0, j + 1) - volume_flux_y(i, j);
        end_volume(i, j) = cell_volume(i, j);
      });
    }

    // DO k=y_lo,y_hi+2
    //   DO j=x_lo,x_hi
    hydro2d::par_ranged2(Span2d{x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2 + 2}, [=] DEVICE_KERNEL(const int x, const int y) {
      int upstream_cell, source_cell, downstream_cell, dif;
      double cell_courant, limiter_coef3, limiter_coef4, courant_dens,  courant_mass, grad_up, grad_down, limit_factor, slope_sign;

      const int j = x;
      const int k = y;

      if (volume_flux_y(x, y) > 0.0) {
        upstream_cell = k - 2;
        source_cell = k - 1;
        downstream_cell = k;
        dif = source_cell;
      } else {
        upstream_cell = min(k + 1, y_hi + 2); // XXX can't do std::min because CUDA
        source_cell = k;
        downstream_cell = k - 1;
        dif = upstream_cell;
      }

      cell_courant = std::fabs(volume_flux_y(x, y)) / start_volume(j, source_cell);
      limiter_coef3 = (1.0 + cell_courant) * (dynode[k] / dynode[dif]);
      limiter_coef4 = 2.0 - cell_courant;

      courant_dens = cell_courant;

      grad_up = density1(j, source_cell) - density1(j, upstream_cell);
      grad_down = density1(j, downstream_cell) - density1(j, source_cell);
      slope_sign = 1.0;
      if (grad_down <= 0.0) slope_sign = -1.0;
      if (grad_up * grad_down > 0.0) {
        limit_factor = (1.0 - courant_dens) * slope_sign *
                  std::fmin(std::fmin(std::fabs(grad_up), std::fabs(grad_down)),
                            sixth * (limiter_coef3 * std::fabs(grad_up) + limiter_coef4 * std::fabs(grad_down)));
      } else {
        limit_factor = 0.0;
      }
      face_mass_flux_y(x, y) = volume_flux_y(x, y) * (density1(j, source_cell) + limit_factor);

      courant_mass = std::fabs(face_mass_flux_y(x, y)) / (density1(j, source_cell) * start_volume(j, source_cell));
      grad_up = spec_energy1(j, source_cell) - spec_energy1(j, upstream_cell);
      grad_down = spec_energy1(j, downstream_cell) - spec_energy1(j, source_cell);
      slope_sign = 1.0;
      if (grad_down <= 0.0) slope_sign = -1.0;
      if (grad_up * grad_down > 0.0) {
        limit_factor = (1.0 - courant_mass) * slope_sign *
                  std::fmin(std::fmin(std::fabs(grad_up), std::fabs(grad_down)),
                            sixth * (limiter_coef3 * std::fabs(grad_up) + limiter_coef4 * std::fabs(grad_down)));
      } else {
        limit_factor = 0.0;
      }
      face_energy_flux(x, y) = face_mass_flux_y(x, y) * (spec_energy1(j, source_cell) + limit_factor);
    });

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi
    hydro2d::par_ranged2(Span2d{x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=] DEVICE_KERNEL(const int i, const int j) {
      double start_mass_s = density1(i, j) * start_volume(i, j);
      double end_mass_s = start_mass_s + face_mass_flux_y(i, j) - face_mass_flux_y(i + 0, j + 1);
      double end_energy_s = (spec_energy1(i, j) * start_mass_s + face_energy_flux(i, j) - face_energy_flux(i + 0, j + 1)) / end_mass_s;
      double remap_volume_s = start_volume(i, j) + volume_flux_y(i, j) - volume_flux_y(i + 0, j + 1);
      density1(i, j) = end_mass_s / remap_volume_s;
      spec_energy1(i, j) = end_energy_s;
    });
  }
}

//  @brief Cell centred flow_sweep_driver solver.
//  @author Wayne Gaudin
//  @details Invokes the user chosen_device flow_sweep_driver kernel.
void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  remap_cell_fields_kernel(t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, sweep_dir, sweep_index, t.field.dxnode, t.field.dynode,
                    t.field.cell_volume, t.field.density1, t.field.spec_energy1, t.field.face_mass_flux_x, t.field.volume_flux_x, t.field.face_mass_flux_y,
                    t.field.volume_flux_y, t.field.scratch_buf1, t.field.scratch_buf2, t.field.scratch_buf3, t.field.scratch_buf4,
                    t.field.scratch_buf5, t.field.scratch_buf6, t.field.scratch_buf7);
}
