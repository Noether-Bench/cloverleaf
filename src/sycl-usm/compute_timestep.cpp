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

#include "compute_timestep.h"
#include "context.h"

//  @brief Fortran compute_dt kernel
//  @author Wayne Gaudin
//  @details Calculates the global_min compute_dt on the mesh subdomain based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.

#define SPLIT

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

  int xStart = x_lo + 1, xEnd = x_hi + 2;
  int yStart = y_lo + 1, yEnd = y_hi + 2;
  int lenX = xEnd - xStart;
  // y = y_lo + 1  |  y_hi + 2

  hydro2d::DeviceArray1D<double> minResults(ctx, 1);

#if defined(__HIPSYCL__) || defined(__OPENSYCL__)
  auto reduction = sycl::reduction(minResults.data, dt_min_val, sycl::global_min<double>());
#else
  auto reduction =
      sycl::reduction(minResults.data, dt_min_val, sycl::global_min<double>(), sycl::property::reduction::initialize_to_identity());
#endif

  ctx.exec_ctx
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for(                                      //
            sycl::range<1>((xEnd - xStart) * (yEnd - yStart)), //
            reduction,                                         //
            [=](sycl::id<1> idx, auto &acc) {
              const auto i = xStart + (idx[0] % lenX);
              const auto j = yStart + (idx[0] / lenX);
              double dsx = dxcell[i];
              double dsy = dycell[j];
              double cc = wave_speed(i, j) * wave_speed(i, j);
              cc = cc + 2.0 * viscosity_a(i, j) / density0(i, j);
              cc = sycl::fmax(sycl::sqrt(cc), glb_tiny);
              double dt_sound = cfl_sound_factor * sycl::fmin(dsx, dsy) / cc;
              double div = 0.0;
              double dv1 = (velx0(i, j) + velx0(i + 0, j + 1)) * xface_area(i, j);
              double dv2 = (velx0(i + 1, j + 0) + velx0(i + 1, j + 1)) * xface_area(i + 1, j + 0);
              div = div + dv2 - dv1;
              double dt_xflow =
                  cfl_ux_factor * 2.0 * cell_volume(i, j) / sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * cell_volume(i, j));

              dv1 = (vely0(i, j) + vely0(i + 1, j + 0)) * yface_area(i, j);
              dv2 = (vely0(i + 0, j + 1) + vely0(i + 1, j + 1)) * yface_area(i + 0, j + 1);
              div = div + dv2 - dv1;
              double dt_yflow =
                  cfl_uy_factor * 2.0 * cell_volume(i, j) / sycl::fmax(sycl::fmax(sycl::fabs(dv1), sycl::fabs(dv2)), glb_tiny * cell_volume(i, j));
              div = div / (2.0 * cell_volume(i, j));
              double dt_divergence;
              if (div < -glb_tiny) {
                dt_divergence = cfl_div_factor * (-1.0 / div);
              } else {
                dt_divergence = glb_huge;
              }
              acc.combine(sycl::fmin(dt_sound, sycl::fmin(dt_xflow, sycl::fmin(dt_yflow, sycl::fmin(dt_divergence, glb_huge)))));
            });
      })
      .wait_and_throw();
  ctx.exec_ctx.wait_and_throw();
  dt_min_val = minResults[0];
  hydro2d::free(ctx.exec_ctx, minResults);

  tile_dt_cause = static_cast<int>(10.01 * (packed_cell_code - static_cast<int>(packed_cell_code)));
  packed_cell_code = packed_cell_code - (packed_cell_code - (int)(packed_cell_code));
  dtmin_j = ((int)packed_cell_code) % x_hi;
  dtmin_k = static_cast<int>(1.f + (packed_cell_code / x_hi));

  if (dt_min_val < dt_floor) too_small = 1;

  if (too_small != 0) {

    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << dtmin_j << " " << dtmin_k << std::endl
              << "x, y                 : " << xcentre[dtmin_j] << " " << ycentre[dtmin_k] << std::endl
              << "compute_dt : " << dt_min_val << std::endl
              << "Cell velocities;" << std::endl
              << velx0(dtmin_j, dtmin_k) << " " << vely0(dtmin_j, dtmin_k) << std::endl
              << velx0(dtmin_j + 1, dtmin_k) << " " << vely0(dtmin_j + 1, dtmin_k) << std::endl
              << velx0(dtmin_j + 1, dtmin_k + 1) << " " << vely0(dtmin_j + 1, dtmin_k + 1) << std::endl
              << velx0(dtmin_j, dtmin_k + 1) << " " << vely0(dtmin_j, dtmin_k + 1) << std::endl
              << "initial_density, initial_energy, cell_pressure, wave_speed " << std::endl
              << density0(dtmin_j, dtmin_k) << " " << energy0(dtmin_j, dtmin_k) << " " << cell_pressure(dtmin_j, dtmin_k) << " " << wave_speed(dtmin_j, dtmin_k)
              << std::endl;
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
