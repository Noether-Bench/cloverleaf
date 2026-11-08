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

//  @brief Fortran compute_dt kernel
//  @author Wayne Gaudin
//  @details Calculates the global_min compute_dt on the mesh subdomain based on the CFL
//  condition, the velocity gradient and the velocity divergence. A safety
//  factor is used to ensure numerical stability.
void compute_timestep_kernel(int x_lo, int x_hi, int y_lo, int y_hi, double dt_floor, double cfl_sound_factor, double cfl_ux_factor, double cfl_uy_factor,
                    double cfl_div_factor, Kokkos::View<double **> &xface_area, Kokkos::View<double **> &yface_area, Kokkos::View<double *> &xcentre,
                    Kokkos::View<double *> &ycentre, Kokkos::View<double *> &dxcell, Kokkos::View<double *> &dycell,
                    Kokkos::View<double **> &cell_volume, Kokkos::View<double **> &density0, Kokkos::View<double **> &energy0,
                    Kokkos::View<double **> &cell_pressure, Kokkos::View<double **> &viscosity_a, Kokkos::View<double **> &wave_speed,
                    Kokkos::View<double **> &velx0, Kokkos::View<double **> &vely0, Kokkos::View<double **> &dt_min, double &dt_min_val,
                    int &tile_dt_cause, double &tile_xcoord, double &tile_ycoord, int &dtmin_j, int &dtmin_k, int &too_small) {

  too_small = 0;
  dt_min_val = glb_huge;
  double packed_cell_code = 1.1;

  // DO k=y_lo,y_hi
  //   DO j=x_lo,x_hi
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});
  Kokkos::parallel_reduce(
      "compute_timestep", policy,
      KOKKOS_LAMBDA(const int j, const int k, double &dt_min_val) {
        double dsx = dxcell(j);
        double dsy = dycell(k);

        double cc = wave_speed(j, k) * wave_speed(j, k);
        cc = cc + 2.0 * viscosity_a(j, k) / density0(j, k);
        cc = Kokkos::max(sqrt(cc), glb_tiny);

        double dt_sound = cfl_sound_factor * Kokkos::min(dsx, dsy) / cc;

        double div = 0.0;

        double dv1 = (velx0(j, k) + velx0(j, k + 1)) * xface_area(j, k);
        double dv2 = (velx0(j + 1, k) + velx0(j + 1, k + 1)) * xface_area(j + 1, k);

        div = div + dv2 - dv1;

        double dt_xflow = cfl_ux_factor * 2.0 * cell_volume(j, k) / Kokkos::max(Kokkos::max(fabs(dv1), fabs(dv2)), glb_tiny * cell_volume(j, k));

        dv1 = (vely0(j, k) + vely0(j + 1, k)) * yface_area(j, k);
        dv2 = (vely0(j, k + 1) + vely0(j + 1, k + 1)) * yface_area(j, k + 1);

        div = div + dv2 - dv1;

        double dt_yflow = cfl_uy_factor * 2.0 * cell_volume(j, k) / Kokkos::max(Kokkos::max(fabs(dv1), fabs(dv2)), glb_tiny * cell_volume(j, k));

        div = div / (2.0 * cell_volume(j, k));

        double dt_divergence;
        if (div < -glb_tiny) {
          dt_divergence = cfl_div_factor * (-1.0 / div);
        } else {
          dt_divergence = glb_huge;
        }

        dt_min_val = Kokkos::min(dt_min_val, dt_sound);
        dt_min_val = Kokkos::min(dt_min_val, dt_xflow);
        dt_min_val = Kokkos::min(dt_min_val, dt_yflow);
        dt_min_val = Kokkos::min(dt_min_val, dt_divergence);
      },
      Kokkos::Min<double>(dt_min_val));

  //  Extract the mimimum compute_dt information
  tile_dt_cause = 10.01 * (packed_cell_code - (int)(packed_cell_code));
  packed_cell_code = packed_cell_code - (packed_cell_code - (int)(packed_cell_code));
  dtmin_j = ((int)packed_cell_code) % x_hi;
  dtmin_k = 1 + (packed_cell_code / x_hi);
  // TODO: cannot do this with GPU memory directly
  // tile_xcoord = xcentre(dtmin_j+1); // Offset by 1 because of Fortran halos in original code
  // tile_ycoord = ycentre(dtmin_k+1);

  if (dt_min_val < dt_floor) too_small = 1;

  if (too_small != 0) {
    std::cout << "Timestep information:" << std::endl
              << "j, k                 : " << dtmin_j << " " << dtmin_k << std::endl
              << "x, y                 : " << xcentre(dtmin_j) << " " << ycentre(dtmin_k) << std::endl
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

  compute_timestep_kernel(globals.subdomain.tile_list[tile_id].meta.tile_xlo, globals.subdomain.tile_list[tile_id].meta.blk_xmax, globals.subdomain.tile_list[tile_id].meta.blk_ymin,
                 globals.subdomain.tile_list[tile_id].meta.blk_ymax, globals.settings.dt_floor, globals.settings.cfl_sound_factor, globals.settings.cfl_ux_factor,
                 globals.settings.cfl_uy_factor, globals.settings.cfl_div_factor, globals.subdomain.tile_list[tile_id].field.xface_area.host_vals,
                 globals.subdomain.tile_list[tile_id].field.yface_area.host_vals, globals.subdomain.tile_list[tile_id].field.xcentre.host_vals,
                 globals.subdomain.tile_list[tile_id].field.ycentre.host_vals, globals.subdomain.tile_list[tile_id].field.dxcell.host_vals,
                 globals.subdomain.tile_list[tile_id].field.dycell.host_vals, globals.subdomain.tile_list[tile_id].field.cell_volume.host_vals,
                 globals.subdomain.tile_list[tile_id].field.density0.host_vals, globals.subdomain.tile_list[tile_id].field.energy0.host_vals,
                 globals.subdomain.tile_list[tile_id].field.cell_pressure.host_vals, globals.subdomain.tile_list[tile_id].field.shock_damping.host_vals,
                 globals.subdomain.tile_list[tile_id].field.wave_speed.host_vals, globals.subdomain.tile_list[tile_id].field.velx0.host_vals,
                 globals.subdomain.tile_list[tile_id].field.vely0.host_vals, globals.subdomain.tile_list[tile_id].field.scratch_buf1.host_vals, tile_dt, limiter_code, tile_xcoord,
                 tile_ycoord, dtmin_j, dtmin_k, too_small);

  if (limiter_code == 1) limiter_name = "sound";
  if (limiter_code == 2) limiter_name = "velx";
  if (limiter_code == 3) limiter_name = "vely";
  if (limiter_code == 4) limiter_name = "div";
}
