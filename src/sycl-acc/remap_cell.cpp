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

//  @brief Fortran cell flow_sweep_driver kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void remap_cell_fields_kernel(exec_ctx &exec_ctx, int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index,
                       hydro2d::DeviceArray1D<double> &vertexdx_buffer, hydro2d::DeviceArray1D<double> &vertexdy_buffer,
                       hydro2d::DeviceArray2D<double> &volume_buffer, hydro2d::DeviceArray2D<double> &local_density1_buf,
                       hydro2d::DeviceArray2D<double> &local_energy1_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf,
                       hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf, hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf,
                       hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf, hydro2d::DeviceArray2D<double> &pre_vol_buffer,
                       hydro2d::DeviceArray2D<double> &post_vol_buffer, hydro2d::DeviceArray2D<double> &pre_mass_buffer,
                       hydro2d::DeviceArray2D<double> &post_mass_buffer, hydro2d::DeviceArray2D<double> &advec_vol_buffer,
                       hydro2d::DeviceArray2D<double> &post_ener_buffer, hydro2d::DeviceArray2D<double> &ener_flux_buffer) {

  const double sixth = 1.0 / 6.0;

  if (dir == glb_dir_x) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2

    const hydro2d::Span2d policy(x_lo - 2 + 1, y_lo - 2 + 1, x_hi + 2 + 2, y_hi + 2 + 2);

    if (sweep_index == 1) {
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_volume = volume_buffer.access<R>(h);
        auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
        auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
        auto start_volume = pre_vol_buffer.access<RW>(h);
        auto end_volume = post_vol_buffer.access<W>(h);
        hydro2d::par_ranged<class advec_cell_xdir_seq1>(h, policy, [=](id<2> idx) {
          start_volume[idx] = cell_volume[idx] + (volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx] + volume_flux_y[hydro2d::offset(idx, 0, 1)] -
                                        volume_flux_y[idx]);
          end_volume[idx] = start_volume[idx] - (volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx]);
        });
      });

    } else {
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_volume = volume_buffer.access<R>(h);
        auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
        auto start_volume = pre_vol_buffer.access<W>(h);
        auto end_volume = post_vol_buffer.access<W>(h);
        hydro2d::par_ranged<class advec_cell_xdir_sne1>(h, policy, [=](id<2> idx) {
          start_volume[idx] = cell_volume[idx] + volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx];
          end_volume[idx] = cell_volume[idx];
        });
      });
    }
    hydro2d::execute(exec_ctx, [&](handler &h) {
      auto dxnode = vertexdx_buffer.access<R>(h);
      auto density1 = local_density1_buf.access<R>(h);
      auto spec_energy1 = local_energy1_buf.access<R>(h);
      auto face_mass_flux_x = local_mass_flux_x_buf.access<RW>(h);
      auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
      auto start_volume = pre_vol_buffer.access<R>(h);
      auto face_energy_flux = ener_flux_buffer.access<W>(h);
      // DO k=y_lo,y_hi
      //   DO j=x_lo,x_hi+2
      hydro2d::par_ranged<class advec_cell_xdir_ener_flux>(h, {x_lo + 1, y_lo + 1, x_hi + 2 + 2, y_hi + 2}, [=](id<2> idx) {
        int upstream_cell, source_cell, downstream_cell, dif;
        double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (volume_flux_x[idx] > 0.0) {
          upstream_cell = j - 2;
          source_cell = j - 1;
          downstream_cell = j;
          dif = source_cell;
        } else {
          upstream_cell = sycl::min(j + 1, x_hi + 2);
          source_cell = j;
          downstream_cell = j - 1;
          dif = upstream_cell;
        }

        cell_courant = sycl::fabs(volume_flux_x[idx]) / start_volume[source_cell][k];
        limiter_coef3 = (1.0 + cell_courant) * (dxnode[j] / dxnode[dif]);
        limiter_coef4 = 2.0 - cell_courant;

        courant_dens = cell_courant;

        grad_up = density1[source_cell][k] - density1[upstream_cell][k];
        grad_down = density1[downstream_cell][k] - density1[source_cell][k];
        slope_sign = 1.0;
        if (grad_down <= 0.0) slope_sign = -1.0;
        if (grad_up * grad_down > 0.0) {
          limit_factor = (1.0 - courant_dens) * slope_sign *
                    sycl::fmin(sycl::fmin(sycl::fabs(grad_up), sycl::fabs(grad_down)),
                               sixth * (limiter_coef3 * sycl::fabs(grad_up) + limiter_coef4 * sycl::fabs(grad_down)));
        } else {
          limit_factor = 0.0;
        }
        face_mass_flux_x[idx] = volume_flux_x[idx] * (density1[source_cell][k] + limit_factor);

        courant_mass = sycl::fabs(face_mass_flux_x[idx]) / (density1[source_cell][k] * start_volume[source_cell][k]);
        grad_up = spec_energy1[source_cell][k] - spec_energy1[upstream_cell][k];
        grad_down = spec_energy1[downstream_cell][k] - spec_energy1[source_cell][k];
        slope_sign = 1.0;
        if (grad_down <= 0.0) slope_sign = -1.0;
        if (grad_up * grad_down > 0.0) {
          limit_factor = (1.0 - courant_mass) * slope_sign *
                    sycl::fmin(sycl::fmin(sycl::fabs(grad_up), sycl::fabs(grad_down)),
                               sixth * (limiter_coef3 * sycl::fabs(grad_up) + limiter_coef4 * sycl::fabs(grad_down)));
        } else {
          limit_factor = 0.0;
        }

        face_energy_flux[idx] = face_mass_flux_x[idx] * (spec_energy1[source_cell][k] + limit_factor);
      });
    });
    hydro2d::execute(exec_ctx, [&](handler &h) {
      auto density1 = local_density1_buf.access<RW>(h);
      auto spec_energy1 = local_energy1_buf.access<RW>(h);
      auto face_mass_flux_x = local_mass_flux_x_buf.access<R>(h);
      auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
      auto start_volume = pre_vol_buffer.access<R>(h);
      auto face_energy_flux = ener_flux_buffer.access<R>(h);
      // DO k=y_lo,y_hi
      //   DO j=x_lo,x_hi

      hydro2d::par_ranged<class advec_cell_xdir_d1e1>(h, {x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=](id<2> idx) {
        double start_mass_s = density1[idx] * start_volume[idx];
        double end_mass_s = start_mass_s + face_mass_flux_x[idx] - face_mass_flux_x[hydro2d::offset(idx, 1, 0)];
        double end_energy_s = (spec_energy1[idx] * start_mass_s + face_energy_flux[idx] - face_energy_flux[hydro2d::offset(idx, 1, 0)]) / end_mass_s;
        double remap_volume_s = start_volume[idx] + volume_flux_x[idx] - volume_flux_x[hydro2d::offset(idx, 1, 0)];
        density1[idx] = end_mass_s / remap_volume_s;
        spec_energy1[idx] = end_energy_s;
      });
    });
  } else if (dir == glb_dir_y) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2
    hydro2d::Span2d policy(x_lo - 2 + 1, y_lo - 2 + 1, x_hi + 2 + 2, y_hi + 2 + 2);

    if (sweep_index == 1) {
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_volume = volume_buffer.access<R>(h);
        auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
        auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
        auto start_volume = pre_vol_buffer.access<RW>(h);
        auto end_volume = post_vol_buffer.access<W>(h);
        hydro2d::par_ranged<class CONCAT_LINE(advec_cell_ydir_s1)>(h, policy, [=](id<2> idx) {
          start_volume[idx] = cell_volume[idx] + (volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx] + volume_flux_x[hydro2d::offset(idx, 1, 0)] -
                                        volume_flux_x[idx]);
          end_volume[idx] = start_volume[idx] - (volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx]);
        });
      });
    } else {
      hydro2d::execute(exec_ctx, [&](handler &h) {
        auto cell_volume = volume_buffer.access<R>(h);
        auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
        auto start_volume = pre_vol_buffer.access<RW>(h);
        auto end_volume = post_vol_buffer.access<W>(h);
        hydro2d::par_ranged<class CONCAT_LINE(advec_cell_ydir_s1)>(h, policy, [=](id<2> idx) {
          start_volume[idx] = cell_volume[idx] + volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx];
          end_volume[idx] = cell_volume[idx];
        });
      });
    }

    hydro2d::execute(exec_ctx, [&](handler &h) {
      auto dynode = vertexdy_buffer.access<R>(h);
      auto density1 = local_density1_buf.access<R>(h);
      auto spec_energy1 = local_energy1_buf.access<R>(h);
      auto face_mass_flux_y = local_mass_flux_y_buf.access<RW>(h);
      auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
      auto start_volume = pre_vol_buffer.access<R>(h);
      auto face_energy_flux = ener_flux_buffer.access<RW>(h);
      // DO k=y_lo,y_hi+2
      //   DO j=x_lo,x_hi
      hydro2d::par_ranged<class advec_cell_ydir_ener_flux>(h, {x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2 + 2}, [=](id<2> idx) {
        int upstream_cell, source_cell, downstream_cell, dif;
        double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (volume_flux_y[idx] > 0.0) {
          upstream_cell = k - 2;
          source_cell = k - 1;
          downstream_cell = k;
          dif = source_cell;
        } else {
          upstream_cell = sycl::min(k + 1, y_hi + 2);
          source_cell = k;
          downstream_cell = k - 1;
          dif = upstream_cell;
        }

        cell_courant = sycl::fabs(volume_flux_y[idx]) / start_volume[j][source_cell];
        limiter_coef3 = (1.0 + cell_courant) * (dynode[k] / dynode[dif]);
        limiter_coef4 = 2.0 - cell_courant;

        courant_dens = cell_courant;

        grad_up = density1[j][source_cell] - density1[j][upstream_cell];
        grad_down = density1[j][downstream_cell] - density1[j][source_cell];
        slope_sign = 1.0;
        if (grad_down <= 0.0) slope_sign = -1.0;
        if (grad_up * grad_down > 0.0) {
          limit_factor = (1.0 - courant_dens) * slope_sign *
                    sycl::fmin(sycl::fmin(sycl::fabs(grad_up), sycl::fabs(grad_down)),
                               sixth * (limiter_coef3 * sycl::fabs(grad_up) + limiter_coef4 * sycl::fabs(grad_down)));
        } else {
          limit_factor = 0.0;
        }
        face_mass_flux_y[idx] = volume_flux_y[idx] * (density1[j][source_cell] + limit_factor);

        courant_mass = sycl::fabs(face_mass_flux_y[idx]) / (density1[j][source_cell] * start_volume[j][source_cell]);
        grad_up = spec_energy1[j][source_cell] - spec_energy1[j][upstream_cell];
        grad_down = spec_energy1[j][downstream_cell] - spec_energy1[j][source_cell];
        slope_sign = 1.0;
        if (grad_down <= 0.0) slope_sign = -1.0;
        if (grad_up * grad_down > 0.0) {
          limit_factor = (1.0 - courant_mass) * slope_sign *
                    sycl::fmin(sycl::fmin(sycl::fabs(grad_up), sycl::fabs(grad_down)),
                               sixth * (limiter_coef3 * sycl::fabs(grad_up) + limiter_coef4 * sycl::fabs(grad_down)));
        } else {
          limit_factor = 0.0;
        }
        face_energy_flux[idx] = face_mass_flux_y[idx] * (spec_energy1[j][source_cell] + limit_factor);
      });
    });

    hydro2d::execute(exec_ctx, [&](handler &h) {
      auto density1 = local_density1_buf.access<RW>(h);
      auto spec_energy1 = local_energy1_buf.access<RW>(h);
      auto face_mass_flux_y = local_mass_flux_y_buf.access<R>(h);
      auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
      auto start_volume = pre_vol_buffer.access<R>(h);
      auto face_energy_flux = ener_flux_buffer.access<R>(h);
      // DO k=y_lo,y_hi
      //   DO j=x_lo,x_hi
      hydro2d::par_ranged<class advec_cell_ydir_e1d1>(h, {x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2}, [=](id<2> idx) {
        double start_mass_s = density1[idx] * start_volume[idx];
        double end_mass_s = start_mass_s + face_mass_flux_y[idx] - face_mass_flux_y[hydro2d::offset(idx, 0, 1)];
        double end_energy_s = (spec_energy1[idx] * start_mass_s + face_energy_flux[idx] - face_energy_flux[hydro2d::offset(idx, 0, 1)]) / end_mass_s;
        double remap_volume_s = start_volume[idx] + volume_flux_y[idx] - volume_flux_y[hydro2d::offset(idx, 0, 1)];
        density1[idx] = end_mass_s / remap_volume_s;
        spec_energy1[idx] = end_energy_s;
      });
    });
  }
}

//  @brief Cell centred flow_sweep_driver solver.
//  @author Wayne Gaudin
//  @details Invokes the user chosen_device flow_sweep_driver kernel.
void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  remap_cell_fields_kernel(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, sweep_dir, sweep_index,
                    t.field.dxnode, t.field.dynode, t.field.cell_volume, t.field.density1, t.field.spec_energy1, t.field.face_mass_flux_x,
                    t.field.volume_flux_x, t.field.face_mass_flux_y, t.field.volume_flux_y, t.field.scratch_buf1, t.field.scratch_buf2,
                    t.field.scratch_buf3, t.field.scratch_buf4, t.field.scratch_buf5, t.field.scratch_buf6, t.field.scratch_buf7);
}
