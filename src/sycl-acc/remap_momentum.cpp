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

//  @brief Fortran momentum flow_sweep_driver kernel
//  @author Wayne Gaudin
//  @details Performs a second order advective remap on the vertex momentum
//  using van-Leer limiting and directional splitting.
//  Note copied although start_volume is only set and not used in the update, please
//  leave it in the method.
void remap_momentum_kernel(exec_ctx &q, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &vel_comp_buffer,
                      hydro2d::DeviceArray2D<double> &local_mass_flux_x_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_x_buf,
                      hydro2d::DeviceArray2D<double> &local_mass_flux_y_buf, hydro2d::DeviceArray2D<double> &local_volume_flux_y_buf,
                      hydro2d::DeviceArray2D<double> &volume_buffer, hydro2d::DeviceArray2D<double> &local_density1_buf,
                      hydro2d::DeviceArray2D<double> &node_flux_buffer, hydro2d::DeviceArray2D<double> &node_mass_post_buffer,
                      hydro2d::DeviceArray2D<double> &node_mass_pre_buffer, hydro2d::DeviceArray2D<double> &mom_flux_buffer,
                      hydro2d::DeviceArray2D<double> &pre_vol_buffer, hydro2d::DeviceArray2D<double> &post_vol_buffer,
                      hydro2d::DeviceArray1D<double> &celldx_buffer, hydro2d::DeviceArray1D<double> &celldy_buffer, int vel_component, int sweep_index,
                      int sweep_dir) {

  int momentum_pass_id = sweep_dir + 2 * (sweep_index - 1);

  // DO k=y_lo-2,y_hi+2
  //   DO j=x_lo-2,x_hi+2

  hydro2d::Span2d policy(x_lo - 2 + 1, y_lo - 2 + 1, x_hi + 2 + 2, y_hi + 2 + 2);

  if (momentum_pass_id == 1) { // x 1
    hydro2d::execute(q, [&](handler &h) {
      auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
      auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
      auto cell_volume = volume_buffer.access<R>(h);
      auto start_volume = pre_vol_buffer.access<W>(h);
      auto end_volume = post_vol_buffer.access<RW>(h);
      hydro2d::par_ranged<class CONCAT_LINE(advec_mom_x1)>(h, policy, [=](id<2> idx) {
        end_volume[idx] = cell_volume[idx] + volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx];
        start_volume[idx] = end_volume[idx] + volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx];
      });
    });
  } else if (momentum_pass_id == 2) { // y 1
    hydro2d::execute(q, [&](handler &h) {
      auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
      auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
      auto cell_volume = volume_buffer.access<R>(h);
      auto start_volume = pre_vol_buffer.access<W>(h);
      auto end_volume = post_vol_buffer.access<RW>(h);
      hydro2d::par_ranged<class CONCAT_LINE(advec_mom_y1)>(h, policy, [=](id<2> idx) {
        end_volume[idx] = cell_volume[idx] + volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx];
        start_volume[idx] = end_volume[idx] + volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx];
      });
    });
  } else if (momentum_pass_id == 3) { // x 2
    hydro2d::execute(q, [&](handler &h) {
      auto volume_flux_y = local_volume_flux_y_buf.access<R>(h);
      auto cell_volume = volume_buffer.access<R>(h);
      auto start_volume = pre_vol_buffer.access<W>(h);
      auto end_volume = post_vol_buffer.access<RW>(h);
      hydro2d::par_ranged<class CONCAT_LINE(advec_mom_x1)>(h, policy, [=](id<2> idx) {
        end_volume[idx] = cell_volume[idx];
        start_volume[idx] = end_volume[idx] + volume_flux_y[hydro2d::offset(idx, 0, 1)] - volume_flux_y[idx];
      });
    });
  } else if (momentum_pass_id == 4) { // y 2
    hydro2d::execute(q, [&](handler &h) {
      auto volume_flux_x = local_volume_flux_x_buf.access<R>(h);
      auto cell_volume = volume_buffer.access<R>(h);
      auto start_volume = pre_vol_buffer.access<W>(h);
      auto end_volume = post_vol_buffer.access<RW>(h);
      hydro2d::par_ranged<class CONCAT_LINE(advec_mom_y1)>(h, policy, [=](id<2> idx) {
        end_volume[idx] = cell_volume[idx];
        start_volume[idx] = end_volume[idx] + volume_flux_x[hydro2d::offset(idx, 1, 0)] - volume_flux_x[idx];
      });
    });
  }

  if (sweep_dir == 1) {
    if (vel_component == 1) {
      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-2,x_hi+2

      hydro2d::execute(q, [&](handler &h) {
        auto face_mass_flux_x = local_mass_flux_x_buf.access<R>(h);
        auto nodal_mass_flux = node_flux_buffer.access<RW>(h);
        hydro2d::par_ranged<class advec_mom_dir1_vel1_node_flux>(
            h, {x_lo - 2 + 1, y_lo + 1, x_hi + 2 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
              // Find staggered mesh mass fluxes, nodal masses and volumes.
              nodal_mass_flux[idx] = 0.25 * (face_mass_flux_x[hydro2d::offset(idx, 0, -1)] + face_mass_flux_x[idx] +
                                       face_mass_flux_x[hydro2d::offset(idx, 1, -1)] + face_mass_flux_x[hydro2d::offset(idx, 1, 0)]);
            });
      });

      // DO k=y_lo,y_hi+1
      //   DO j=x_lo-1,x_hi+2

      hydro2d::execute(q, [&](handler &h) {
        auto density1 = local_density1_buf.access<R>(h);
        auto nodal_mass_flux = node_flux_buffer.access<R>(h);
        auto nodal_mass_after = node_mass_post_buffer.access<RW>(h);
        auto nodal_mass_before = node_mass_pre_buffer.access<RW>(h);
        auto end_volume = post_vol_buffer.access<R>(h);
        hydro2d::par_ranged<class advec_mom_dir1_vel1_node_mass_pre>(
            h, {x_lo - 1 + 1, y_lo + 1, x_hi + 2 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
              // Staggered cell mass post flow_sweep_driver
              nodal_mass_after[idx] =
                  0.25 * (density1[hydro2d::offset(idx, 0, -1)] * end_volume[hydro2d::offset(idx, 0, -1)] + density1[idx] * end_volume[idx] +
                          density1[hydro2d::offset(idx, -1, -1)] * end_volume[hydro2d::offset(idx, -1, -1)] +
                          density1[hydro2d::offset(idx, -1, 0)] * end_volume[hydro2d::offset(idx, -1, 0)]);
              nodal_mass_before[idx] = nodal_mass_after[idx] - nodal_mass_flux[hydro2d::offset(idx, -1, 0)] + nodal_mass_flux[idx];
            });
      });
    }

    // DO k=y_lo,y_hi+1
    //  DO j=x_lo-1,x_hi+1

    hydro2d::execute(q, [&](handler &h) {
      auto vel_comp = vel_comp_buffer.access<R>(h);
      auto nodal_mass_flux = node_flux_buffer.access<R>(h);
      auto nodal_mass_before = node_mass_pre_buffer.access<R>(h);
      auto momentum_flux = mom_flux_buffer.access<RW>(h);
      auto dxcell = celldx_buffer.access<R>(h);
      hydro2d::par_ranged<class advec_mom_dir1_mom_flux>(h, {x_lo - 1 + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
        int upstream_cell, source_cell, downstream_cell, dif;
        double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (nodal_mass_flux[idx] < 0.0) {
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

        node_courant = sycl::fabs(nodal_mass_flux[idx]) / (nodal_mass_before[source_cell][k]);
        cell_width = dxcell[j];
        vel_jump_up = vel_comp[source_cell][k] - vel_comp[upstream_cell][k];
        vel_jump_down = vel_comp[downstream_cell][k] - vel_comp[source_cell][k];
        limit_factor = 0.0;
        if (vel_jump_up * vel_jump_down > 0.0) {
          auw = sycl::fabs(vel_jump_up);
          adw = sycl::fabs(vel_jump_down);
          slope_sign = 1.0;
          if (vel_jump_down <= 0.0) slope_sign = -1.0;
          limit_factor =
              slope_sign * sycl::fmin(sycl::fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dxcell[dif]) / 6.0, auw), adw);
        }
        remap_face_vel = vel_comp[source_cell][k] + (1.0 - node_courant) * limit_factor;
        momentum_flux[idx] = remap_face_vel * nodal_mass_flux[idx];
      });
    });

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    hydro2d::execute(q, [&](handler &h) {
      auto vel_comp = vel_comp_buffer.access<RW>(h);
      auto nodal_mass_after = node_mass_post_buffer.access<R>(h);
      auto nodal_mass_before = node_mass_pre_buffer.access<R>(h);
      auto momentum_flux = mom_flux_buffer.access<R>(h);
      hydro2d::par_ranged<class advec_mom_dir1_vel1>(h, {x_lo + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
        vel_comp[idx] = (vel_comp[idx] * nodal_mass_before[idx] + momentum_flux[hydro2d::offset(idx, -1, 0)] - momentum_flux[idx]) / nodal_mass_after[idx];
      });
    });
  } else if (sweep_dir == 2) {
    if (vel_component == 1) {
      // DO k=y_lo-2,y_hi+2
      //   DO j=x_lo,x_hi+1

      hydro2d::execute(q, [&](handler &h) {
        auto nodal_mass_flux = node_flux_buffer.access<RW>(h);
        auto face_mass_flux_y = local_mass_flux_y_buf.access<R>(h);
        hydro2d::par_ranged<class advec_mom_dir2_vel1_node_flux>(
            h, {x_lo + 1, y_lo - 2 + 1, x_hi + 1 + 2, y_hi + 2 + 2}, [=](id<2> idx) {
              // Find staggered mesh mass fluxes and nodal masses and volumes.
              nodal_mass_flux[idx] = 0.25 * (face_mass_flux_y[hydro2d::offset(idx, -1, 0)] + face_mass_flux_y[idx] +
                                       face_mass_flux_y[hydro2d::offset(idx, -1, 1)] + face_mass_flux_y[hydro2d::offset(idx, 0, 1)]);
            });
      });

      // DO k=y_lo-1,y_hi+2
      //   DO j=x_lo,x_hi+1

      hydro2d::execute(q, [&](handler &h) {
        auto density1 = local_density1_buf.access<R>(h);
        auto nodal_mass_flux = node_flux_buffer.access<R>(h);
        auto nodal_mass_after = node_mass_post_buffer.access<RW>(h);
        auto nodal_mass_before = node_mass_pre_buffer.access<RW>(h);
        auto end_volume = post_vol_buffer.access<R>(h);
        hydro2d::par_ranged<class advec_mom_dir2_vel1_node_mass_pre>(
            h, {x_lo + 1, y_lo - 1 + 1, x_hi + 1 + 2, y_hi + 2 + 2}, [=](id<2> idx) {
              nodal_mass_after[idx] =
                  0.25 * (density1[hydro2d::offset(idx, 0, -1)] * end_volume[hydro2d::offset(idx, 0, -1)] + density1[idx] * end_volume[idx] +
                          density1[hydro2d::offset(idx, -1, -1)] * end_volume[hydro2d::offset(idx, -1, -1)] +
                          density1[hydro2d::offset(idx, -1, 0)] * end_volume[hydro2d::offset(idx, -1, 0)]);
              nodal_mass_before[idx] = nodal_mass_after[idx] - nodal_mass_flux[hydro2d::offset(idx, 0, -1)] + nodal_mass_flux[idx];
            });
      });
    }

    // DO k=y_lo-1,y_hi+1
    //   DO j=x_lo,x_hi+1

    hydro2d::execute(q, [&](handler &h) {
      auto vel_comp = vel_comp_buffer.access<R>(h);
      auto nodal_mass_flux = node_flux_buffer.access<R>(h);
      auto nodal_mass_before = node_mass_pre_buffer.access<R>(h);
      auto momentum_flux = mom_flux_buffer.access<RW>(h);
      auto dycell = celldy_buffer.access<R>(h);
      hydro2d::par_ranged<class advec_mom_dir2_mom_flux>(h, {x_lo + 1, y_lo - 1 + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
        int upstream_cell, source_cell, downstream_cell, dif;
        double node_courant, cell_width, limit_factor, vel_jump_up, vel_jump_down, auw, adw, slope_sign, remap_face_vel;

        const int j = idx.get(0);
        const int k = idx.get(1);

        if (nodal_mass_flux[idx] < 0.0) {
          upstream_cell = k + 2;
          source_cell = k + 1;
          downstream_cell = k;
          dif = source_cell;
        } else {
          upstream_cell = k - 1;
          source_cell = k;
          downstream_cell = k + 1;
          dif = upstream_cell;
        }

        node_courant = sycl::fabs(nodal_mass_flux[idx]) / (nodal_mass_before[j][source_cell]);
        cell_width = dycell[k];
        vel_jump_up = vel_comp[j][source_cell] - vel_comp[j][upstream_cell];
        vel_jump_down = vel_comp[j][downstream_cell] - vel_comp[j][source_cell];
        limit_factor = 0.0;
        if (vel_jump_up * vel_jump_down > 0.0) {
          auw = sycl::fabs(vel_jump_up);
          adw = sycl::fabs(vel_jump_down);
          slope_sign = 1.0;
          if (vel_jump_down <= 0.0) slope_sign = -1.0;
          limit_factor =
              slope_sign * sycl::fmin(sycl::fmin(cell_width * ((2.0 - node_courant) * adw / cell_width + (1.0 + node_courant) * auw / dycell[dif]) / 6.0, auw), adw);
        }
        remap_face_vel = vel_comp[j][source_cell] + (1.0 - node_courant) * limit_factor;
        momentum_flux[idx] = remap_face_vel * nodal_mass_flux[idx];
      });
    });

    // DO k=y_lo,y_hi+1
    //   DO j=x_lo,x_hi+1

    hydro2d::execute(q, [&](handler &h) {
      auto vel_comp = vel_comp_buffer.access<RW>(h);
      auto nodal_mass_after = node_mass_post_buffer.access<R>(h);
      auto nodal_mass_before = node_mass_pre_buffer.access<R>(h);
      auto momentum_flux = mom_flux_buffer.access<R>(h);
      hydro2d::par_ranged<class advec_mom_dir2_vel1>(h, {x_lo + 1, y_lo + 1, x_hi + 1 + 2, y_hi + 1 + 2}, [=](id<2> idx) {
        vel_comp[idx] = (vel_comp[idx] * nodal_mass_before[idx] + momentum_flux[hydro2d::offset(idx, 0, -1)] - momentum_flux[idx]) / nodal_mass_after[idx];
      });
    });
  }
}

//  @brief Momentum flow_sweep_driver solver
//  @author Wayne Gaudin
//  @details Invokes the user specified momentum flow_sweep_driver kernel.
void remap_momentum_driver(sim_state &globals, int tile_id, int vel_component, int sweep_dir, int sweep_index) {

  tile_block &t = globals.subdomain.tile_list[tile_id];
  if (vel_component == 1) {
    remap_momentum_kernel(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                     t.field.velx1,       //
                     t.field.face_mass_flux_x, //
                     t.field.volume_flux_x,  //
                     t.field.face_mass_flux_y, //
                     t.field.volume_flux_y,  //
                     t.field.cell_volume,      //
                     t.field.density1,    //
                     t.field.scratch_buf1, //
                     t.field.scratch_buf2, //
                     t.field.scratch_buf3, //
                     t.field.scratch_buf4, //
                     t.field.scratch_buf5, //
                     t.field.scratch_buf6, //
                     t.field.dxcell,      //
                     t.field.dycell,      //
                     vel_component, sweep_index, sweep_dir);
  } else {
    remap_momentum_kernel(globals.context.exec_ctx, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax,
                     t.field.vely1,       //
                     t.field.face_mass_flux_x, //
                     t.field.volume_flux_x,  //
                     t.field.face_mass_flux_y, //
                     t.field.volume_flux_y,  //
                     t.field.cell_volume,      //
                     t.field.density1,    //
                     t.field.scratch_buf1, //
                     t.field.scratch_buf2, //
                     t.field.scratch_buf3, //
                     t.field.scratch_buf4, //
                     t.field.scratch_buf5, //
                     t.field.scratch_buf6, //
                     t.field.dxcell,      //
                     t.field.dycell,      //
                     vel_component, sweep_index, sweep_dir);
  }
}
