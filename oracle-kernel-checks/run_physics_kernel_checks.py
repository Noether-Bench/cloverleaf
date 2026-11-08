#!/usr/bin/env python3
"""Compile and run focused Hydro2dMain physics kernel oracle checks."""

from __future__ import annotations

import os
import re
import json
import subprocess
import sys
from pathlib import Path


HARNESS = r'''
#include "PdV.h"
#include "advance_velocity.h"
#include "remap_cell.h"
#include "remap_momentum.h"
#include "compute_timestep.h"
#include "compute_face_fluxes.h"
#include "populate_subdomain.h"
#include "equation_of_state.h"
#include "setup_subdomain_mesh.h"
#include "roll_time_level.h"
#include "rollback_state.h"
#include "shock_damping.h"

#include <array>
#include <cmath>
#include <iostream>
#include <string>

void advance_velocity_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field);
void remap_cell_fields_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, int dir, int sweep_index, field_set &field);
void remap_momentum_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &vel_comp_buffer,
                      field_set &field, int vel_component, int sweep_index, int sweep_dir);
void compute_timestep_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, double dt_floor, double cfl_sound_factor, double cfl_ux_factor,
                    double cfl_uy_factor, double cfl_div_factor, field_set &field, double &dt_min_val, int &tile_dt_cause, double &tile_xcoord,
                    double &tile_ycoord, int &dtmin_j, int &dtmin_k, int &too_small);
void compute_face_fluxes_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field);
void equation_of_state_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field,
                      hydro2d::DeviceArray2D<double> &density_in, hydro2d::DeviceArray2D<double> &energy_in);
void compression_work_kernel(bool use_target, bool predictor_pass, int x_lo, int x_hi, int y_lo, int y_hi, double dt, field_set &field);
void roll_time_level_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field);
void rollback_state_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field);
void shock_damping_kernel(bool use_target, int x_lo, int x_hi, int y_lo, int y_hi, field_set &field);

static int failures = 0;
static int checks_seen = 0;
static std::string active_check;

static size_t at(const hydro2d::DeviceArray2D<double> &b, int i, int j) {
  return static_cast<size_t>(i) + static_cast<size_t>(j) * b.nX();
}

static void check_close(const std::string &label, double got, double expected, double tol = 1.0e-10) {
  if (!active_check.empty() && label != active_check) return;
  ++checks_seen;
  if (std::fabs(got - expected) > tol) {
    std::cerr << label << ": got " << got << " expected " << expected << "\n";
    ++failures;
  }
}

static void check_true(const std::string &label, bool ok) {
  if (!active_check.empty() && label != active_check) return;
  ++checks_seen;
  if (!ok) {
    std::cerr << label << ": condition failed\n";
    ++failures;
  }
}

static void fill2(hydro2d::DeviceArray2D<double> &b, double local_val) {
  for (size_t i = 0; i < b.N(); ++i) b.data[i] = local_val;
}

static void fill1(hydro2d::DeviceArray1D<double> &b, double local_val) {
  for (size_t i = 0; i < b.N(); ++i) b.data[i] = local_val;
}

static region_spec region_id(double initial_density, double initial_energy, double velx, double vely, region_shape_kind shape_selector,
                        double x_lower, double y_lower, double x_upper = 0.0, double y_upper = 0.0, double circle_radius = 0.0) {
  region_spec s{};
  s.defined = true;
  s.initial_density = initial_density;
  s.initial_energy = initial_energy;
  s.velx = velx;
  s.vely = vely;
  s.shape_selector = shape_selector;
  s.x_lower = x_lower;
  s.y_lower = y_lower;
  s.x_upper = x_upper;
  s.y_upper = y_upper;
  s.circle_radius = circle_radius;
  return s;
}

static sim_state make_globals(hydro2d::context &ctx) {
  run_config cfg{};
  cfg.host_staging = true;
  cfg.tiles_per_subdomain = 1;
  cfg.total_subdomains = 1;
  cfg.region_count = 4;
  cfg.dt_initial = 0.1;
  cfg.dt_floor = 1.0e-8;
  cfg.cfl_sound_factor = 0.5;
  cfg.cfl_ux_factor = 0.6;
  cfg.cfl_uy_factor = 0.7;
  cfg.cfl_div_factor = 0.8;
  cfg.timing_enabled = false;
  cfg.mesh_spec.x_lower = 0.0;
  cfg.mesh_spec.y_lower = 0.0;
  cfg.mesh_spec.x_upper = 4.0;
  cfg.mesh_spec.y_upper = 4.0;
  cfg.mesh_spec.grid_nx = 4;
  cfg.mesh_spec.grid_ny = 4;
  cfg.regions.push_back(region_id(0.2, 1.0, 0.1, -0.2, glb_rectangle, 0.0, 0.0, 4.0, 4.0));
  cfg.regions.push_back(region_id(1.1, 2.0, -0.3, 0.4, glb_rectangle, 0.5, 0.5, 2.5, 2.5));
  cfg.regions.push_back(region_id(1.7, 3.0, 0.5, 0.6, glb_circle, 3.5, 3.5, 0.0, 0.0, 0.4));
  cfg.regions.push_back(region_id(2.3, 4.0, -0.7, -0.8, glb_single_point, 0.0, 0.0));

  std::array<int, 4> neighbours = {mesh_edge, mesh_edge, mesh_edge, mesh_edge};
  subdomain_type subdomain(neighbours, 0, 1, 1, 4, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1);
  tile_desc meta{};
  meta.tile_adjacent = {tile_edge, tile_edge, tile_edge, tile_edge};
  meta.edge_mask = {1, 1, 1, 1};
  meta.tile_xlo = 1;
  meta.blk_xmax = 4;
  meta.blk_ymin = 1;
  meta.blk_ymax = 4;
  meta.tile_west_idx = 1;
  meta.right_idx = 4;
  meta.blk_bottom = 1;
  meta.blk_top = 4;
  subdomain.tile_list.emplace_back(meta, ctx);
  return sim_state(cfg, ctx, std::move(subdomain));
}

static void initialise_regular_field(field_set &f) {
  fill2(f.cell_volume, 2.0);
  fill2(f.xface_area, 1.5);
  fill2(f.yface_area, 2.5);
  fill2(f.density0, 2.0);
  fill2(f.density1, 2.0);
  fill2(f.energy0, 3.0);
  fill2(f.spec_energy1, 3.0);
  fill2(f.cell_pressure, 5.0);
  fill2(f.shock_damping, 0.0);
  fill2(f.wave_speed, 1.0);
  fill2(f.velx0, 0.4);
  fill2(f.vely0, -0.2);
  fill2(f.velx1, 0.4);
  fill2(f.vely1, -0.2);
  fill2(f.volume_flux_x, 0.1);
  fill2(f.volume_flux_y, -0.08);
  fill2(f.face_mass_flux_x, 0.2);
  fill2(f.face_mass_flux_y, -0.16);
  fill2(f.scratch_buf1, 0.0);
  fill2(f.scratch_buf2, 0.0);
  fill2(f.scratch_buf3, 0.0);
  fill2(f.scratch_buf4, 0.0);
  fill2(f.scratch_buf5, 0.0);
  fill2(f.scratch_buf6, 0.0);
  fill2(f.scratch_buf7, 0.0);
  fill1(f.dxcell, 1.0);
  fill1(f.dycell, 1.0);
  fill1(f.dxnode, 1.0);
  fill1(f.dynode, 1.0);
}

static void test_initialise_and_generate() {
  hydro2d::context ctx{false};
  sim_state g = make_globals(ctx);
  setup_subdomain_mesh(0, g);
  field_set &f = g.subdomain.tile_list[0].field;

  check_close("startup-xnode", f.xnode[3], 1.0);
  check_close("startup-xcentre", f.xcentre[3], 1.5);
  check_close("startup-cell_volume", f.cell_volume.data[at(f.cell_volume, 3, 3)], 1.0);
  check_close("startup-xface_area", f.xface_area.data[at(f.xface_area, 3, 3)], 1.0);
  check_close("startup-yface_area", f.yface_area.data[at(f.yface_area, 3, 3)], 1.0);

  populate_subdomain(0, g);
  check_close("generate-background-initial_density", f.density0.data[at(f.density0, 1, 1)], 0.2);
  check_close("generate-rectangle-initial_density", f.density0.data[at(f.density0, 3, 3)], 1.1);
  check_close("generate-circle-initial_energy", f.energy0.data[at(f.energy0, 5, 5)], 3.0);
  check_close("generate-point-initial_density", f.density0.data[at(f.density0, 2, 2)], 2.3);
  check_close("generate-point-velx", f.velx0.data[at(f.velx0, 2, 2)], -0.7);
}

static void test_ideal_gas() {
  hydro2d::context ctx{false};
  field_set f(10, 10, ctx);
  initialise_regular_field(f);
  f.density0.data[at(f.density0, 3, 3)] = 2.5;
  f.energy0.data[at(f.energy0, 3, 3)] = 4.0;
  equation_of_state_kernel(false, 2, 5, 2, 5, f, f.density0, f.energy0);
  check_close("ideal-gas-cell_pressure", f.cell_pressure.data[at(f.cell_pressure, 3, 3)], 4.0);
  check_close("ideal-gas-wave_speed", f.wave_speed.data[at(f.wave_speed, 3, 3)], std::sqrt(2.24));
}

static void test_accelerate_and_flux() {
  hydro2d::context ctx{false};
  field_set f(10, 10, ctx);
  initialise_regular_field(f);
  const int i = 3, j = 3;
  f.density0.data[at(f.density0, i - 1, j - 1)] = 2.0;
  f.density0.data[at(f.density0, i - 1, j)] = 2.0;
  f.density0.data[at(f.density0, i, j)] = 2.0;
  f.density0.data[at(f.density0, i, j - 1)] = 2.0;
  f.cell_volume.data[at(f.cell_volume, i - 1, j - 1)] = 2.0;
  f.cell_volume.data[at(f.cell_volume, i - 1, j)] = 2.0;
  f.cell_volume.data[at(f.cell_volume, i, j)] = 2.0;
  f.cell_volume.data[at(f.cell_volume, i, j - 1)] = 2.0;
  f.cell_pressure.data[at(f.cell_pressure, i, j)] = 9.0;
  f.cell_pressure.data[at(f.cell_pressure, i - 1, j)] = 5.0;
  f.cell_pressure.data[at(f.cell_pressure, i, j - 1)] = 7.0;
  f.cell_pressure.data[at(f.cell_pressure, i - 1, j - 1)] = 4.0;
  f.shock_damping.data[at(f.shock_damping, i, j)] = 1.5;
  f.shock_damping.data[at(f.shock_damping, i - 1, j)] = 0.5;
  f.shock_damping.data[at(f.shock_damping, i, j - 1)] = 1.0;
  f.shock_damping.data[at(f.shock_damping, i - 1, j - 1)] = 0.25;
  f.xface_area.data[at(f.xface_area, i, j)] = 1.0;
  f.xface_area.data[at(f.xface_area, i, j - 1)] = 2.0;
  f.yface_area.data[at(f.yface_area, i, j)] = 1.5;
  f.yface_area.data[at(f.yface_area, i - 1, j)] = 0.5;
  f.velx0.data[at(f.velx0, i, j)] = 0.7;
  f.vely0.data[at(f.vely0, i, j)] = -0.4;

  advance_velocity_kernel(false, 2, 5, 2, 5, 0.2, f);
  const double cycle_index = 0.1 / 4.0;
  const double expected_x = 0.7 - cycle_index * (1.0 * (9.0 - 5.0) + 2.0 * (7.0 - 4.0))
                                - cycle_index * (1.0 * (1.5 - 0.5) + 2.0 * (1.0 - 0.25));
  const double expected_y = -0.4 - cycle_index * (1.5 * (9.0 - 7.0) + 0.5 * (5.0 - 4.0))
                                  - cycle_index * (1.5 * (1.5 - 1.0) + 0.5 * (0.5 - 0.25));
  check_close("advance_velocity-velx", f.velx1.data[at(f.velx1, i, j)], expected_x);
  check_close("advance_velocity-vely", f.vely1.data[at(f.vely1, i, j)], expected_y);

  f.velx0.data[at(f.velx0, i, j)] = 1.0;
  f.velx0.data[at(f.velx0, i, j + 1)] = 2.0;
  f.velx1.data[at(f.velx1, i, j)] = 3.0;
  f.velx1.data[at(f.velx1, i, j + 1)] = 4.0;
  f.vely0.data[at(f.vely0, i, j)] = -1.0;
  f.vely0.data[at(f.vely0, i + 1, j)] = -2.0;
  f.vely1.data[at(f.vely1, i, j)] = -3.0;
  f.vely1.data[at(f.vely1, i + 1, j)] = -4.0;
  f.xface_area.data[at(f.xface_area, i, j)] = 2.0;
  f.yface_area.data[at(f.yface_area, i, j)] = 3.0;
  compute_face_fluxes_kernel(false, 2, 5, 2, 5, 0.1, f);
  check_close("flux_time-calc-x", f.volume_flux_x.data[at(f.volume_flux_x, i, j)], 0.5);
  check_close("flux_time-calc-y", f.volume_flux_y.data[at(f.volume_flux_y, i, j)], -0.75);
}

static void test_pdv_reset_revert() {
  hydro2d::context ctx{false};
  field_set f(10, 10, ctx);
  initialise_regular_field(f);
  const int i = 3, j = 3;
  f.cell_volume.data[at(f.cell_volume, i, j)] = 10.0;
  f.density0.data[at(f.density0, i, j)] = 2.0;
  f.energy0.data[at(f.energy0, i, j)] = 5.0;
  f.cell_pressure.data[at(f.cell_pressure, i, j)] = 4.0;
  f.shock_damping.data[at(f.shock_damping, i, j)] = 2.0;
  f.xface_area.data[at(f.xface_area, i, j)] = 1.0;
  f.xface_area.data[at(f.xface_area, i + 1, j)] = 1.0;
  f.yface_area.data[at(f.yface_area, i, j)] = 1.0;
  f.yface_area.data[at(f.yface_area, i, j + 1)] = 1.0;
  f.velx0.data[at(f.velx0, i, j)] = 1.0;
  f.velx0.data[at(f.velx0, i, j + 1)] = 1.0;
  f.velx0.data[at(f.velx0, i + 1, j)] = 3.0;
  f.velx0.data[at(f.velx0, i + 1, j + 1)] = 3.0;
  f.vely0.data[at(f.vely0, i, j)] = 2.0;
  f.vely0.data[at(f.vely0, i + 1, j)] = 2.0;
  f.vely0.data[at(f.vely0, i, j + 1)] = 4.0;
  f.vely0.data[at(f.vely0, i + 1, j + 1)] = 4.0;

  compression_work_kernel(false, true, 2, 5, 2, 5, 0.1, f);
  const double net_face_flux = 0.2;
  check_close("pdv-predictor_pass-initial_energy", f.spec_energy1.data[at(f.spec_energy1, i, j)], 5.0 - ((4.0 / 2.0 + 2.0 / 2.0) * net_face_flux / 10.0));
  check_close("pdv-predictor_pass-initial_density", f.density1.data[at(f.density1, i, j)], 2.0 * 10.0 / 10.2);

  f.density1.data[at(f.density1, i, j)] = 8.0;
  f.spec_energy1.data[at(f.spec_energy1, i, j)] = 9.0;
  rollback_state_kernel(false, 2, 5, 2, 5, f);
  check_close("rollback_state-initial_density", f.density1.data[at(f.density1, i, j)], 2.0);
  check_close("rollback_state-initial_energy", f.spec_energy1.data[at(f.spec_energy1, i, j)], 5.0);

  f.density1.data[at(f.density1, i, j)] = 6.0;
  f.spec_energy1.data[at(f.spec_energy1, i, j)] = 7.0;
  f.velx1.data[at(f.velx1, i, j)] = 8.0;
  f.vely1.data[at(f.vely1, i, j)] = 9.0;
  roll_time_level_kernel(false, 2, 5, 2, 5, f);
  check_close("reset_time-initial_density", f.density0.data[at(f.density0, i, j)], 6.0);
  check_close("reset_time-initial_energy", f.energy0.data[at(f.energy0, i, j)], 7.0);
  check_close("reset_time-velx", f.velx0.data[at(f.velx0, i, j)], 8.0);
  check_close("reset_time-vely", f.vely0.data[at(f.vely0, i, j)], 9.0);
}

static void test_calc_dt_and_viscosity() {
  hydro2d::context ctx{false};
  field_set f(10, 10, ctx);
  initialise_regular_field(f);
  fill2(f.wave_speed, 1.0);
  f.wave_speed.data[at(f.wave_speed, 3, 3)] = 10.0;
  double dt = 0.0, xl = 0.0, yl = 0.0;
  int control = 0, dtmin_j = 0, dtmin_k = 0, too_small = 0;
  compute_timestep_kernel(false, 2, 5, 2, 5, 1.0e-8, 0.5, 0.6, 0.7, 0.8, f, dt, control, xl, yl, dtmin_j, dtmin_k, too_small);
  check_close("calc-dt-sound-limit", dt, 0.05);
  check_true("calc-dt-not-too_small", too_small == 0);

  fill2(f.shock_damping, -1.0);
  fill2(f.cell_pressure, 1.0);
  f.cell_pressure.data[at(f.cell_pressure, 2, 3)] = 1.0;
  f.cell_pressure.data[at(f.cell_pressure, 4, 3)] = 3.0;
  f.cell_pressure.data[at(f.cell_pressure, 3, 2)] = 1.0;
  f.cell_pressure.data[at(f.cell_pressure, 3, 4)] = 3.0;
  f.velx0.data[at(f.velx0, 3, 3)] = 2.0;
  f.velx0.data[at(f.velx0, 3, 4)] = 2.0;
  f.velx0.data[at(f.velx0, 4, 3)] = 0.0;
  f.velx0.data[at(f.velx0, 4, 4)] = 0.0;
  f.vely0.data[at(f.vely0, 3, 3)] = 2.0;
  f.vely0.data[at(f.vely0, 4, 3)] = 2.0;
  f.vely0.data[at(f.vely0, 3, 4)] = 0.0;
  f.vely0.data[at(f.vely0, 4, 4)] = 0.0;
  shock_damping_kernel(false, 2, 5, 2, 5, f);
  check_true("shock_damping-compression-positive", f.shock_damping.data[at(f.shock_damping, 3, 3)] > 0.0);

  f.velx0.data[at(f.velx0, 3, 3)] = 0.0;
  f.velx0.data[at(f.velx0, 3, 4)] = 0.0;
  f.velx0.data[at(f.velx0, 4, 3)] = 2.0;
  f.velx0.data[at(f.velx0, 4, 4)] = 2.0;
  f.vely0.data[at(f.vely0, 3, 3)] = 0.0;
  f.vely0.data[at(f.vely0, 4, 3)] = 0.0;
  f.vely0.data[at(f.vely0, 3, 4)] = 2.0;
  f.vely0.data[at(f.vely0, 4, 4)] = 2.0;
  shock_damping_kernel(false, 2, 5, 2, 5, f);
  check_close("shock_damping-expansion-zero", f.shock_damping.data[at(f.shock_damping, 3, 3)], 0.0);
}

static void test_advection() {
  hydro2d::context ctx{false};
  field_set f(12, 12, ctx);
  initialise_regular_field(f);
  fill2(f.cell_volume, 10.0);
  fill2(f.density1, 2.0);
  fill2(f.spec_energy1, 3.0);
  fill2(f.volume_flux_x, 0.2);
  fill2(f.volume_flux_y, -0.15);

  remap_cell_fields_kernel(false, 3, 6, 3, 6, glb_dir_x, 1, f);
  check_close("advec-cell-x-initial_density", f.density1.data[at(f.density1, 4, 4)], 2.0);
  check_close("advec-cell-x-initial_energy", f.spec_energy1.data[at(f.spec_energy1, 4, 4)], 3.0);
  check_close("advec-cell-x-mass-flux_time", f.face_mass_flux_x.data[at(f.face_mass_flux_x, 4, 4)], 0.4);

  remap_cell_fields_kernel(false, 3, 6, 3, 6, glb_dir_y, 2, f);
  check_close("advec-cell-y-initial_density", f.density1.data[at(f.density1, 4, 4)], 2.0);
  check_close("advec-cell-y-initial_energy", f.spec_energy1.data[at(f.spec_energy1, 4, 4)], 3.0);
  check_close("advec-cell-y-mass-flux_time", f.face_mass_flux_y.data[at(f.face_mass_flux_y, 4, 4)], -0.3);

  fill2(f.velx1, 4.0);
  fill2(f.vely1, -5.0);
  fill2(f.face_mass_flux_x, 0.4);
  fill2(f.face_mass_flux_y, -0.3);
  remap_momentum_kernel(false, 3, 6, 3, 6, f.velx1, f, 1, 1, glb_dir_x);
  remap_momentum_kernel(false, 3, 6, 3, 6, f.vely1, f, 2, 1, glb_dir_x);
  check_close("advec-mom-x-velx", f.velx1.data[at(f.velx1, 4, 4)], 4.0);
  check_close("advec-mom-x-vely", f.vely1.data[at(f.vely1, 4, 4)], -5.0);
  remap_momentum_kernel(false, 3, 6, 3, 6, f.velx1, f, 1, 2, glb_dir_y);
  remap_momentum_kernel(false, 3, 6, 3, 6, f.vely1, f, 2, 2, glb_dir_y);
  check_close("advec-mom-y-velx", f.velx1.data[at(f.velx1, 4, 4)], 4.0);
  check_close("advec-mom-y-vely", f.vely1.data[at(f.vely1, 4, 4)], -5.0);
}

void fatal_message(char *, char *) {}
void refresh_ghost_cells(sim_state &, int[FIELD_COUNT], int) {}
void hydro2d_max_error(int &) {}
double wall_seconds() { return 0.0; }

int main(int arg_count, char **arg_values) {
  if (arg_count == 3 && std::string(arg_values[1]) == "--check") {
    active_check = arg_values[2];
  } else if (arg_count != 1) {
    std::cerr << "usage: " << arg_values[0] << " [--check CHECK_NAME]\n";
    return 2;
  }

  if (active_check.empty() || active_check.rfind("startup-", 0) == 0 || active_check.rfind("generate-", 0) == 0) {
    test_initialise_and_generate();
  }
  if (active_check.empty() || active_check.rfind("ideal-gas-", 0) == 0) {
    test_ideal_gas();
  }
  if (active_check.empty() || active_check.rfind("advance_velocity-", 0) == 0 || active_check.rfind("flux_time-calc-", 0) == 0) {
    test_accelerate_and_flux();
  }
  if (active_check.empty() || active_check.rfind("pdv-", 0) == 0 || active_check.rfind("rollback_state-", 0) == 0 ||
      active_check.rfind("reset_time-", 0) == 0) {
    test_pdv_reset_revert();
  }
  if (active_check.empty() || active_check.rfind("calc-dt-", 0) == 0 || active_check.rfind("shock_damping-", 0) == 0) {
    test_calc_dt_and_viscosity();
  }
  if (active_check.empty() || active_check.rfind("advec-cell-", 0) == 0 || active_check.rfind("advec-mom-", 0) == 0) {
    test_advection();
  }

  if (!active_check.empty() && checks_seen == 0) {
    std::cerr << "unknown check: " << active_check << "\n";
    return 2;
  }
  if (failures != 0) {
    if (active_check.empty()) {
      std::cerr << failures << " physics kernel checks failed\n";
    } else {
      std::cerr << "Hydro2dMain physics kernel check failed: " << active_check << "\n";
    }
    return 1;
  }
  if (active_check.empty()) {
    std::cout << "Hydro2dMain physics kernel checks passed\n";
  } else {
    std::cout << "Hydro2dMain physics kernel check passed: " << active_check << "\n";
  }
  return 0;
}
'''


def cmake_cache_value(cache: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}(?::[^=]+)?=(.*)$")
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def check_names() -> list[str]:
    return re.findall(r'\bcheck_(?:close|true|unchanged)\s*\(\s*"([^"]+)"', HARNESS)


def parse_failures(stderr: str) -> dict[str, str]:
    names = set(check_names())
    failures: dict[str, str] = {}
    for line in stderr.splitlines():
        label, sep, detail = line.partition(": ")
        if sep and label in names:
            failures[label] = detail
    return failures


def build_harness(home_proc: Path) -> Path:
    cache = home_proc / "build/eval/CMakeCache.txt"
    if not cache.exists():
        print(f"missing CMake cache: {cache}", file=sys.stderr)
        raise SystemExit(2)

    compiler = cmake_cache_value(cache, "CMAKE_CXX_COMPILER") or os.environ.get("CXX") or "c++"
    # These focused harnesses call the kernels with use_target=false and check
    # semantics on ordinary host buffers. Compiling them without device_offload flags
    # avoids NVHPC/OpenMP target runtime startup and teardown instability while
    # the app regression suite still verifies real GPU execution.
    flags: list[str] = []

    build_dir = home_proc / "build/kernel-checks"
    build_dir.mkdir(parents=True, exist_ok=True)
    harness = build_dir / "physics_kernel_checks.cpp"
    harness.write_text(HARNESS, encoding="utf-8")
    exe = build_dir / "physics_kernel_checks"

    cmd = [
        compiler,
        "-std=c++17",
        "-DNO_MPI",
        "-DUSE_OMP_TARGET",
        "-DOMP_TARGET",
        "-DDIFFUSE_OVERLOAD",
        "-I",
        "solver",
        "-I",
        "src/omp-target",
        str(harness),
        "solver/data_types.cpp",
        "src/omp-target/advance_velocity.cpp",
        "src/omp-target/remap_cell.cpp",
        "src/omp-target/remap_momentum.cpp",
        "src/omp-target/compute_timestep.cpp",
        "src/omp-target/compute_face_fluxes.cpp",
        "src/omp-target/populate_subdomain.cpp",
        "src/omp-target/equation_of_state.cpp",
        "src/omp-target/setup_subdomain_mesh.cpp",
        "src/omp-target/PdV.cpp",
        "src/omp-target/roll_time_level.cpp",
        "src/omp-target/rollback_state.cpp",
        "src/omp-target/shock_damping.cpp",
        "-o",
        str(exe),
        *flags,
    ]
    proc = subprocess.run(cmd, cwd=home_proc, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)

    return exe


def collect_results(home_proc: Path, path: Path) -> int:
    exe = home_proc / "build/kernel-checks/physics_kernel_checks"
    if not exe.exists():
        exe = build_harness(home_proc)

    env = os.environ.copy()
    env.pop("OMP_TARGET_OFFLOAD", None)
    proc = subprocess.run([str(exe)], cwd=home_proc, text=True, capture_output=True, env=env)
    failures = parse_failures(proc.stderr)
    results = {}
    if proc.returncode in (0, 1):
        for label in check_names():
            if label in failures:
                results[label] = {"status": "fail", "detail": failures[label]}
            else:
                results[label] = {"status": "pass", "detail": ""}
    else:
        for label in check_names():
            results[label] = {
                "status": "error",
                "detail": f"harness exited with {proc.returncode}",
            }

    output = {
        "harness": "physics-kernel-semantics",
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if proc.returncode == 0:
        print("Hydro2dMain physics kernel checks passed")
        return 0
    if proc.returncode == 1 and failures:
        print(f"recorded {len(failures)} Hydro2dMain physics kernel check failures")
        return 0
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    return 1


def check_result(path: Path, label: str) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("results", {}).get(label)
    if result is None:
        print(f"unknown check result: {label}", file=sys.stderr)
        return 2
    if result.get("status") == "pass":
        print(f"Hydro2dMain physics kernel check passed: {label}")
        return 0
    print(f"{label}: {result.get('detail', '')}", file=sys.stderr)
    print(f"Hydro2dMain physics kernel check failed: {label}", file=sys.stderr)
    return 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check")
    parser.add_argument("--collect-results")
    parser.add_argument("--check-result", nargs=2, metavar=("RESULTS_JSON", "CHECK_NAME"))
    parser.add_argument("--list-checks", action="store_true")
    cli_args = parser.parse_args()

    if cli_args.list_checks:
        for label in check_names():
            print(label)
        return 0

    home_proc = Path.cwd()
    if cli_args.check_result:
        return check_result(Path(cli_args.check_result[0]), cli_args.check_result[1])
    if cli_args.collect_results:
        return collect_results(home_proc, Path(cli_args.collect_results))

    exe = home_proc / "build/kernel-checks/physics_kernel_checks"
    if cli_args.build_only or not exe.exists():
        exe = build_harness(home_proc)
    if cli_args.build_only:
        return 0

    command = [str(exe)]
    if cli_args.check:
        command += ["--check", cli_args.check]
    env = os.environ.copy()
    env.pop("OMP_TARGET_OFFLOAD", None)
    run = subprocess.run(command, cwd=home_proc, text=True, capture_output=True, env=env)
    print(run.stdout, end="")
    print(run.stderr, end="", file=sys.stderr)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
