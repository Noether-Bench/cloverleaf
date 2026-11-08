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

//  @brief Fortran cell flow_sweep_driver kernel.
//  @author Wayne Gaudin
//  @details Performs a second order advective remap using van-Leer limiting
//  with directional splitting.
void remap_cell_fields_kernel(int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, Kokkos::View<double *> &dxnode,
                       Kokkos::View<double *> &dynode, Kokkos::View<double **> &cell_volume, Kokkos::View<double **> &density1,
                       Kokkos::View<double **> &spec_energy1, Kokkos::View<double **> &face_mass_flux_x, Kokkos::View<double **> &volume_flux_x,
                       Kokkos::View<double **> &face_mass_flux_y, Kokkos::View<double **> &volume_flux_y, Kokkos::View<double **> &start_volume,
                       Kokkos::View<double **> &end_volume, Kokkos::View<double **> &pre_mass, Kokkos::View<double **> &post_mass,
                       Kokkos::View<double **> &advec_vol, Kokkos::View<double **> &post_ener, Kokkos::View<double **> &face_energy_flux) {

  const double sixth = 1.0 / 6.0;

  if (dir == glb_dir_x) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo - 2 + 1, y_lo - 2 + 1}, {x_hi + 2 + 2, y_hi + 2 + 2});

    if (sweep_index == 1) {
      Kokkos::parallel_for(
          "remap_cell xdir sweep_index=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            start_volume(j, k) = cell_volume(j, k) + (volume_flux_x(j + 1, k) - volume_flux_x(j, k) + volume_flux_y(j, k + 1) - volume_flux_y(j, k));
            end_volume(j, k) = start_volume(j, k) - (volume_flux_x(j + 1, k) - volume_flux_x(j, k));
          });
    } else {
      Kokkos::parallel_for(
          "remap_cell xdir sweep_index!=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            start_volume(j, k) = cell_volume(j, k) + volume_flux_x(j + 1, k) - volume_flux_x(j, k);
            end_volume(j, k) = cell_volume(j, k);
          });
    }

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_x2({x_lo + 1, y_lo + 1}, {x_hi + 2 + 2, y_hi + 2});
    Kokkos::parallel_for(
        "remap_cell xdir face_energy_flux", policy_x2, KOKKOS_LAMBDA(const int j, const int k) {
          int upstream_cell, source_cell, downstream_cell, dif;
          double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;

          if (volume_flux_x(j, k) > 0.0) {
            upstream_cell = j - 2;
            source_cell = j - 1;
            downstream_cell = j;
            dif = source_cell;
          } else {
            upstream_cell = Kokkos::min(j + 1, x_hi + 2);
            source_cell = j;
            downstream_cell = j - 1;
            dif = upstream_cell;
          }

          cell_courant = fabs(volume_flux_x(j, k)) / start_volume(source_cell, k);
          limiter_coef3 = (1.0 + cell_courant) * (dxnode(j) / dxnode(dif));
          limiter_coef4 = 2.0 - cell_courant;

          courant_dens = cell_courant;

          grad_up = density1(source_cell, k) - density1(upstream_cell, k);
          grad_down = density1(downstream_cell, k) - density1(source_cell, k);
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_dens) * slope_sign *
                      Kokkos::min(Kokkos::min(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_mass_flux_x(j, k) = volume_flux_x(j, k) * (density1(source_cell, k) + limit_factor);

          courant_mass = fabs(face_mass_flux_x(j, k)) / (density1(source_cell, k) * start_volume(source_cell, k));
          grad_up = spec_energy1(source_cell, k) - spec_energy1(upstream_cell, k);
          grad_down = spec_energy1(downstream_cell, k) - spec_energy1(source_cell, k);
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_mass) * slope_sign *
                      Kokkos::min(Kokkos::min(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }

          face_energy_flux(j, k) = face_mass_flux_x(j, k) * (spec_energy1(source_cell, k) + limit_factor);
        });

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_xy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});
    Kokkos::parallel_for(
        "remap_cell xdir density1,spec_energy1", policy_xy, KOKKOS_LAMBDA(const int j, const int k) {
          double start_mass_s = density1(j, k) * start_volume(j, k);
          double end_mass_s = start_mass_s + face_mass_flux_x(j, k) - face_mass_flux_x(j + 1, k);
          double end_energy_s = (spec_energy1(j, k) * start_mass_s + face_energy_flux(j, k) - face_energy_flux(j + 1, k)) / end_mass_s;
          double remap_volume_s = start_volume(j, k) + volume_flux_x(j, k) - volume_flux_x(j + 1, k);
          density1(j, k) = end_mass_s / remap_volume_s;
          spec_energy1(j, k) = end_energy_s;
        });
  }

  else if (dir == glb_dir_y) {

    // DO k=y_lo-2,y_hi+2
    //   DO j=x_lo-2,x_hi+2
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({x_lo - 2 + 1, y_lo - 2 + 1}, {x_hi + 2 + 2, y_hi + 2 + 2});

    if (sweep_index == 1) {
      Kokkos::parallel_for(
          "remap_cell ydir sweep_index=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            start_volume(j, k) = cell_volume(j, k) + (volume_flux_y(j, k + 1) - volume_flux_y(j, k) + volume_flux_x(j + 1, k) - volume_flux_x(j, k));
            end_volume(j, k) = start_volume(j, k) - (volume_flux_y(j, k + 1) - volume_flux_y(j, k));
          });
    } else {
      Kokkos::parallel_for(
          "remap_cell ydir sweep_index!=1", policy, KOKKOS_LAMBDA(const int j, const int k) {
            start_volume(j, k) = cell_volume(j, k) + volume_flux_y(j, k + 1) - volume_flux_y(j, k);
            end_volume(j, k) = cell_volume(j, k);
          });
    }

    // DO k=y_lo,y_hi+2
    //   DO j=x_lo,x_hi
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_y2({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2 + 2});
    Kokkos::parallel_for(
        "remap_cell ydir face_energy_flux", policy_y2, KOKKOS_LAMBDA(const int j, const int k) {
          int upstream_cell, source_cell, downstream_cell, dif;
          double cell_courant, limiter_coef3, limiter_coef4, courant_dens, courant_mass, grad_up, grad_down, limit_factor, slope_sign;

          if (volume_flux_y(j, k) > 0.0) {
            upstream_cell = k - 2;
            source_cell = k - 1;
            downstream_cell = k;
            dif = source_cell;
          } else {
            upstream_cell = Kokkos::min(k + 1, y_hi + 2);
            source_cell = k;
            downstream_cell = k - 1;
            dif = upstream_cell;
          }

          cell_courant = fabs(volume_flux_y(j, k)) / start_volume(j, source_cell);
          limiter_coef3 = (1.0 + cell_courant) * (dynode(k) / dynode(dif));
          limiter_coef4 = 2.0 - cell_courant;

          courant_dens = cell_courant;

          grad_up = density1(j, source_cell) - density1(j, upstream_cell);
          grad_down = density1(j, downstream_cell) - density1(j, source_cell);
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_dens) * slope_sign *
                      Kokkos::min(Kokkos::min(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_mass_flux_y(j, k) = volume_flux_y(j, k) * (density1(j, source_cell) + limit_factor);

          courant_mass = fabs(face_mass_flux_y(j, k)) / (density1(j, source_cell) * start_volume(j, source_cell));
          grad_up = spec_energy1(j, source_cell) - spec_energy1(j, upstream_cell);
          grad_down = spec_energy1(j, downstream_cell) - spec_energy1(j, source_cell);
          slope_sign = 1.0;
          if (grad_down <= 0.0) slope_sign = -1.0;
          if (grad_up * grad_down > 0.0) {
            limit_factor = (1.0 - courant_mass) * slope_sign *
                      Kokkos::min(Kokkos::min(fabs(grad_up), fabs(grad_down)), sixth * (limiter_coef3 * fabs(grad_up) + limiter_coef4 * fabs(grad_down)));
          } else {
            limit_factor = 0.0;
          }
          face_energy_flux(j, k) = face_mass_flux_y(j, k) * (spec_energy1(j, source_cell) + limit_factor);
        });

    // DO k=y_lo,y_hi
    //   DO j=x_lo,x_hi
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy_xy({x_lo + 1, y_lo + 1}, {x_hi + 2, y_hi + 2});
    Kokkos::parallel_for(
        "remap_cell ydir density1,spec_energy1", policy_xy, KOKKOS_LAMBDA(const int j, const int k) {
          double start_mass_s = density1(j, k) * start_volume(j, k);
          double end_mass_s = start_mass_s + face_mass_flux_y(j, k) - face_mass_flux_y(j, k + 1);
          double end_energy_s = (spec_energy1(j, k) * start_mass_s + face_energy_flux(j, k) - face_energy_flux(j, k + 1)) / end_mass_s;
          double remap_volume_s = start_volume(j, k) + volume_flux_y(j, k) - volume_flux_y(j, k + 1);
          density1(j, k) = end_mass_s / remap_volume_s;
          spec_energy1(j, k) = end_energy_s;
        });
  }
}

//  @brief Cell centred flow_sweep_driver solver.
//  @author Wayne Gaudin
//  @details Invokes the user chosen_device flow_sweep_driver kernel.
void remap_cell_fields_driver(sim_state &globals, int tile_id, int sweep_index, int sweep_dir) {

  remap_cell_fields_kernel(globals.subdomain.tile_list[tile_id].meta.tile_xlo, globals.subdomain.tile_list[tile_id].meta.blk_xmax, globals.subdomain.tile_list[tile_id].meta.blk_ymin,
                    globals.subdomain.tile_list[tile_id].meta.blk_ymax, sweep_dir, sweep_index, globals.subdomain.tile_list[tile_id].field.dxnode.host_vals,
                    globals.subdomain.tile_list[tile_id].field.dynode.host_vals, globals.subdomain.tile_list[tile_id].field.cell_volume.host_vals,
                    globals.subdomain.tile_list[tile_id].field.density1.host_vals, globals.subdomain.tile_list[tile_id].field.spec_energy1.host_vals,
                    globals.subdomain.tile_list[tile_id].field.face_mass_flux_x.host_vals, globals.subdomain.tile_list[tile_id].field.volume_flux_x.host_vals,
                    globals.subdomain.tile_list[tile_id].field.face_mass_flux_y.host_vals, globals.subdomain.tile_list[tile_id].field.volume_flux_y.host_vals,
                    globals.subdomain.tile_list[tile_id].field.scratch_buf1.host_vals, globals.subdomain.tile_list[tile_id].field.scratch_buf2.host_vals,
                    globals.subdomain.tile_list[tile_id].field.scratch_buf3.host_vals, globals.subdomain.tile_list[tile_id].field.scratch_buf4.host_vals,
                    globals.subdomain.tile_list[tile_id].field.scratch_buf5.host_vals, globals.subdomain.tile_list[tile_id].field.scratch_buf6.host_vals,
                    globals.subdomain.tile_list[tile_id].field.scratch_buf7.host_vals);
}
