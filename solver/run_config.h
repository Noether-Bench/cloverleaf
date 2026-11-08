#pragma once

#include <string>
#include <vector>

struct mesh_spec_type {

  double x_lower;
  double y_lower;
  double x_upper;
  double y_upper;

  int grid_nx;
  int grid_ny;
};

enum region_shape_kind { glb_rectangle = 1, glb_circle = 2, glb_single_point = 3 };

struct region_spec {

  bool defined;

  double initial_density;
  double initial_energy;
  double velx;
  double vely;

  region_shape_kind shape_selector;

  double x_lower;
  double y_lower;
  double x_upper;
  double y_upper;
  double circle_radius;
};

struct run_config {
  std::string dumpPath;
  bool host_staging;
  std::vector<region_spec> regions;
  int region_count;
  int tiles_per_subdomain;
  int test_case_id;
  bool timing_enabled;
  double stop_time;
  int max_steps;

  double dt_initial;
  double dt_floor;
  double dt_ceiling;
  double dt_growth;
  double cfl_ux_factor;
  double cfl_uy_factor;
  double cfl_sound_factor;
  double cfl_div_factor;
  double dtc;
  double dtu;
  double dtv;
  double divergence_safety;

  int viz_interval;
  int summary_interval;
  int total_subdomains;

  mesh_spec_type mesh_spec;
};
