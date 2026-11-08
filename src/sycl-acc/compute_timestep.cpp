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

#include <cmath>

#include "compute_timestep.h"
#include "context.h"
#include "sycl_reduction.hpp"

//  @brief Fortran compute_dt kernel
//  @author Wayne Gaudin
//  @details Calculates the global_min compute_dt on the mesh subdomain based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.

// #define USE_SYCL2020_REDUCTION

void compute_timestep_kernel(hydro2d::context &ctx, int x_lo, int x_hi, int y_lo, int y_hi, double dt_floor, double cfl_sound_factor, double cfl_ux_factor,
                    double cfl_uy_factor, double cfl_div_factor, hydro2d::DeviceArray2D<double> xface_area, hydro2d::DeviceArray2D<double> yface_area,
                    hydro2d::DeviceArray1D<double> xcentre, hydro2d::DeviceArray1D<double> ycentre, hydro2d::DeviceArray1D<double> dxcell,
                    hydro2d::DeviceArray1D<double> dycell, hydro2d::DeviceArray2D<double> cell_volume, hydro2d::DeviceArray2D<double> density0,
                    hydro2d::DeviceArray2D<double> energy0, hydro2d::DeviceArray2D<double> cell_pressure, hydro2d::DeviceArray2D<double> viscosity_a,
                    hydro2d::DeviceArray2D<double> wave_speed, hydro2d::DeviceArray2D<double> velx0, hydro2d::DeviceArray2D<double> vely0, double &dt_min_val,
                    int &tile_dt_cause, double &tile_xcoord, double &tile_ycoord, int &dtmin_j, int &dtmin_k, int &too_small) {

  too_small = 0;
  dt_min_val = glb_huge;
  double packed_cell_code = 1.1;

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  //	Kokkos::MDRangePolicy <Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});

  auto policy = hydro2d::Span2d(x_lo + 1, y_lo + 1, x_hi + 2, y_hi + 2);

#ifdef USE_SYCL2020_REDUCTION
  hydro2d::DeviceArray1D<double> minResults(ctx, 1);
  minResults.access()[0] = dt_min_val;

  hydro2d::execute(ctx.exec_ctx, [&](handler &h) {
    auto xarea_ = xface_area.access<R>(h);
    auto yarea_ = yface_area.access<R>(h);
    auto celldx_ = dxcell.access<R>(h);
    auto celldy_ = dycell.access<R>(h);
    auto volume_ = cell_volume.access<R>(h);
    auto density0_ = density0.access<R>(h);
    auto viscosity_a_ = viscosity_a.access<R>(h);
    auto soundspeed_ = wave_speed.access<R>(h);
    auto xvel0_ = velx0.access<R>(h);
    auto yvel0_ = vely0.access<R>(h);

  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    h.parallel_for(                                                                     //
        sycl::range<2>(sycl::range<2>(policy.lenX, policy.lenY)),                     //
        sycl::reduction(minResults.access<RW>(h), dt_min_val, sycl::global_min<double>()), //
        [=](sycl::item<2> idxNoOffset, auto &acc) {
          const auto idx = hydro2d::offset(idxNoOffset, policy.startX, policy.startY);
  #else
    // FIXME maxThreadPerBlock = N with nd_range launch is a workaround for https://github.com/intel/llvm/issues/8414
    //  A normal non-nd_range launch blows the register budget as the thread-per-block is passed directly to CUDA PI.
    //  It's unclear how this workaround would affect moved_from platforms.
    size_t maxThreadPerBlock = 256;
    size_t localX = std::ceil(double(policy.lenX) / double(maxThreadPerBlock));
    size_t localY = std::ceil(double(policy.lenY) / double(maxThreadPerBlock));

    auto uniformLocalX = policy.lenX % localX == 0 ? localX : policy.lenX + (localX - policy.lenX % localX);
    auto uniformLocalY = policy.lenY % localY == 0 ? localY : policy.lenY + (localY - policy.lenY % localY);
    uniformLocalX = uniformLocalX >= policy.lenX ? 1 : uniformLocalX;
    uniformLocalY = uniformLocalY >= policy.lenY ? 1 : uniformLocalY;
    h.parallel_for(                                                                                                  //
        sycl::nd_range<2>(sycl::range<2>(policy.lenX, policy.lenY), sycl::range<2>(uniformLocalX, uniformLocalY)), //
        sycl::reduction(minResults.field_buf, h, dt_min_val, sycl::global_min<>(),
                        sycl::property::reduction::initialize_to_identity()), //
        [=](sycl::nd_item<2> idxNoOffset, auto &acc) {
          const auto idx = hydro2d::offset(idxNoOffset.get_global_id(), policy.startX, policy.startY);
  #endif
          double dsx = celldx_[idx[0]];
          double dsy = celldy_[idx[1]];

          double cc = soundspeed_[idx] * soundspeed_[idx];
          cc = cc + 2.0 * viscosity_a_[idx] / density0_[idx];
          cc = sycl::fmax(sycl::sqrt(cc), glb_tiny);

          double dt_sound = cfl_sound_factor * sycl::fmin(dsx, dsy) / cc;

          double div = 0.0;

          double dv1 = (xvel0_[idx] + xvel0_[hydro2d::offset(idx, 0, 1)]) * xarea_[idx];
          double dv2 = (xvel0_[hydro2d::offset(idx, 1, 0)] + xvel0_[hydro2d::offset(idx, 1, 1)]) * xarea_[hydro2d::offset(idx, 1, 0)];

          div = div + dv2 - dv1;

          double dt_xflow = cfl_ux_factor * 2.0 * volume_[idx] / sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * volume_[idx]);

          dv1 = (yvel0_[idx] + yvel0_[hydro2d::offset(idx, 1, 0)]) * yarea_[idx];
          dv2 = (yvel0_[hydro2d::offset(idx, 0, 1)] + yvel0_[hydro2d::offset(idx, 1, 1)]) * yarea_[hydro2d::offset(idx, 0, 1)];

          div = div + dv2 - dv1;

          double dt_yflow = cfl_uy_factor * 2.0 * volume_[idx] / sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * volume_[idx]);

          div = div / (2.0 * volume_[idx]);

          double dt_divergence;
          if (div < -glb_tiny) {
            dt_divergence = cfl_div_factor * (-1.0 / div);
          } else {
            dt_divergence = glb_huge;
          }
          acc.combine(sycl::fmin(dt_sound, sycl::fmin(dt_xflow, sycl::fmin(dt_yflow, sycl::fmin(dt_divergence, glb_huge)))));
        });
  });
  dt_min_val = minResults.access()[0];
#else
  struct captures {
    hydro2d::Accessor<double, 2, R>::Type xface_area;
    hydro2d::Accessor<double, 2, R>::Type yface_area;
    hydro2d::Accessor<double, 1, R>::Type dxcell;
    hydro2d::Accessor<double, 1, R>::Type dycell;
    hydro2d::Accessor<double, 2, R>::Type cell_volume;
    hydro2d::Accessor<double, 2, R>::Type density0;
    hydro2d::Accessor<double, 2, R>::Type viscosity_a;
    hydro2d::Accessor<double, 2, R>::Type wave_speed;
    hydro2d::Accessor<double, 2, R>::Type velx0;
    hydro2d::Accessor<double, 2, R>::Type vely0;
  };

  using Reducer = hydro2d::local_reducer<double, double, captures>;

  hydro2d::DeviceArray1D<double> result(ctx, policy.lenX * policy.lenY);

  hydro2d::par_reduce_2d<class dt_kernel_reduce, double>(
      ctx.exec_ctx, policy,
      [=](handler &h, size_t &size) mutable {
        return Reducer(h, size,
                       {xface_area.access<R>(h), yface_area.access<R>(h), dxcell.access<R>(h), dycell.access<R>(h), cell_volume.access<R>(h),
                        density0.access<R>(h), viscosity_a.access<R>(h), wave_speed.access<R>(h), velx0.access<R>(h), vely0.access<R>(h)},
                       result.field_buf);
      },
      [](const Reducer &ctx, id<1> lidx) { ctx.line_copy[lidx] = glb_huge; },
      [cfl_sound_factor, cfl_uy_factor, cfl_ux_factor, cfl_div_factor](Reducer ctx, id<1> lidx, id<2> idx) {
        double dsx = ctx.elems.dxcell[idx[0]];
        double dsy = ctx.elems.dycell[idx[1]];

        double cc = ctx.elems.wave_speed[idx] * ctx.elems.wave_speed[idx];
        cc = cc + 2.0 * ctx.elems.viscosity_a[idx] / ctx.elems.density0[idx];
        cc = sycl::fmax(sycl::sqrt(cc), glb_tiny);

        double dt_sound = cfl_sound_factor * sycl::fmin(dsx, dsy) / cc;

        double div = 0.0;

        double dv1 = (ctx.elems.velx0[idx] + ctx.elems.velx0[hydro2d::offset(idx, 0, 1)]) * ctx.elems.xface_area[idx];
        double dv2 = (ctx.elems.velx0[hydro2d::offset(idx, 1, 0)] + ctx.elems.velx0[hydro2d::offset(idx, 1, 1)]) *
                     ctx.elems.xface_area[hydro2d::offset(idx, 1, 0)];

        div = div + dv2 - dv1;

        double dt_xflow = cfl_ux_factor * 2.0 * ctx.elems.cell_volume[idx] /
                      sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * ctx.elems.cell_volume[idx]);

        dv1 = (ctx.elems.vely0[idx] + ctx.elems.vely0[hydro2d::offset(idx, 1, 0)]) * ctx.elems.yface_area[idx];
        dv2 = (ctx.elems.vely0[hydro2d::offset(idx, 0, 1)] + ctx.elems.vely0[hydro2d::offset(idx, 1, 1)]) *
              ctx.elems.yface_area[hydro2d::offset(idx, 0, 1)];

        div = div + dv2 - dv1;

        double dt_yflow = cfl_uy_factor * 2.0 * ctx.elems.cell_volume[idx] /
                      sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * ctx.elems.cell_volume[idx]);

        div = div / (2.0 * ctx.elems.cell_volume[idx]);

        double dt_divergence;
        if (div < -glb_tiny) {
          dt_divergence = cfl_div_factor * (-1.0 / div);
        } else {
          dt_divergence = glb_huge;
        }

        double cell_dt_min = sycl::fmin(dt_sound, sycl::fmin(dt_xflow, sycl::fmin(dt_yflow, sycl::fmin(dt_divergence, glb_huge))));
        ctx.line_copy[lidx] = sycl::fmin(ctx.line_copy[lidx], cell_dt_min);
      },
      [](const Reducer &ctx, id<1> idx, id<1> idy) { ctx.line_copy[idx] = sycl::fmin(ctx.line_copy[idx], ctx.line_copy[idy]); },
      [](const Reducer &ctx, size_t group, id<1> idx) { ctx.result[group] = ctx.line_copy[idx]; });

  dt_min_val = result.access()[0];

#endif

  //  Extract the mimimum compute_dt information
  tile_dt_cause = static_cast<int>(10.01 * (packed_cell_code - static_cast<int>(packed_cell_code)));
  packed_cell_code = packed_cell_code - (packed_cell_code - (int)(packed_cell_code));
  dtmin_j = ((int)packed_cell_code) % x_hi;
  dtmin_k = static_cast<int>(1.f + (packed_cell_code / x_hi));
  // TODO: cannot do this with GPU memory directly
  // tile_xcoord = xcentre(dtmin_j+1); // Offset by 1 because of Fortran halos in original code
  // tile_ycoord = ycentre(dtmin_k+1);

  if (dt_min_val < dt_floor) too_small = 1;

  if (too_small != 0) {

    auto xcentre_view = xcentre.access();
    auto ycentre_view = ycentre.access();
    auto density0_view = density0.access();
    auto energy0_view = energy0.access();
    auto pressure_view = cell_pressure.access();
    auto soundspeed_view = wave_speed.access();
    auto velx0_view = velx0.access();
    auto vely0_view = vely0.access();

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << dtmin_j << " " << dtmin_k << std::endl
              << "x, y                 : " << xcentre_view[dtmin_j] << " " << ycentre_view[dtmin_k] << std::endl
              << "compute_dt : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << velx0_view[dtmin_j][dtmin_k] << " " << vely0_view[dtmin_j][dtmin_k] << std::endl
              << velx0_view[dtmin_j + 1][dtmin_k] << " " << vely0_view[dtmin_j + 1][dtmin_k] << std::endl
              << velx0_view[dtmin_j + 1][dtmin_k + 1] << " " << vely0_view[dtmin_j + 1][dtmin_k + 1] << std::endl
              << velx0_view[dtmin_j][dtmin_k + 1] << " " << vely0_view[dtmin_j][dtmin_k + 1] << std::endl
              << "initial_density, initial_energy, cell_pressure, wave_speed " << std::endl
              << density0_view[dtmin_j][dtmin_k] << " " << energy0_view[dtmin_j][dtmin_k] << " " << pressure_view[dtmin_j][dtmin_k] << " "
              << soundspeed_view[dtmin_j][dtmin_k] << std::endl;
  }
}

//  @brief Driver for the compute_dt kernels
//  @author Wayne Gaudin
//  @details Invokes the user specified compute_dt kernel.
void compute_timestep(sim_state &globals, int tile_id, double &tile_dt, std::string &limiter_name, double &tile_xcoord, double &tile_ycoord, int &dtmin_j,
             int &dtmin_k) {

  tile_dt = glb_huge;

  int limiter_code;
  int too_small = 0;

  tile_block &t = globals.subdomain.tile_list[tile_id];
  compute_timestep_kernel(globals.context, t.meta.tile_xlo, t.meta.blk_xmax, t.meta.blk_ymin, t.meta.blk_ymax, globals.settings.dt_floor, globals.settings.cfl_sound_factor,
                 globals.settings.cfl_ux_factor, globals.settings.cfl_uy_factor, globals.settings.cfl_div_factor, t.field.xface_area, t.field.yface_area, t.field.xcentre,
                 t.field.ycentre, t.field.dxcell, t.field.dycell, t.field.cell_volume, t.field.density0, t.field.energy0, t.field.cell_pressure,
                 t.field.shock_damping, t.field.wave_speed, t.field.velx0, t.field.vely0, tile_dt, limiter_code, tile_xcoord, tile_ycoord, dtmin_j, dtmin_k,
                 too_small);

  if (limiter_code == 1) limiter_name = "sound";
  if (limiter_code == 2) limiter_name = "velx";
  if (limiter_code == 3) limiter_name = "vely";
  if (limiter_code == 4) limiter_name = "div";
}
