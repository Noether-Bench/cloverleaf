

#include "input_deck.h"

#include "run_log.h"

#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>

extern std::ostream glb_log;

void parse_deck(std::ifstream &glb_input, mpi_context &parallel, run_config &globals) {

  globals.test_case_id = 0;

  int highest_region = 0;

  globals.mesh_spec.x_lower = 0.0;
  globals.mesh_spec.y_lower = 0.0;
  globals.mesh_spec.x_upper = 0.0;
  globals.mesh_spec.y_upper = 0.0;

  globals.mesh_spec.grid_nx = 10;
  globals.mesh_spec.grid_ny = 10;

  globals.stop_time = 10.0;
  globals.max_steps = glb_huge_int;

  globals.viz_interval = 0;
  globals.summary_interval = 10;

  globals.tiles_per_subdomain = 1;

  globals.dt_initial = 0.1;
  globals.dt_ceiling = 1.0;
  globals.dt_floor = 0.0000001;
  globals.dt_growth = 1.5;
  globals.cfl_sound_factor = 0.7;
  globals.cfl_ux_factor = 0.5;
  globals.cfl_uy_factor = 0.5;
  globals.cfl_div_factor = 0.7;

  globals.timing_enabled = false;


  if (parallel.is_root) {
    glb_log << "Reading input file" << std::endl << std::endl;
  }

  std::string line;

  while (true) {
    std::getline(glb_input, line);
    if (glb_input.eof()) break;
    if (line.empty()) continue;

    std::istringstream iss(line);
    std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
    if (tokens[0] != "region_id") continue;
    highest_region = std::max(highest_region, std::stoi(tokens[1]));
  }

  globals.region_count = highest_region;

  if (globals.region_count < 1) fatal_message((char *)"parse_deck", (char *)"No regions defined.");

  globals.regions = std::vector<region_spec>(globals.region_count);
  for (int s = 0; s < globals.region_count; ++s) {
    globals.regions[s].defined = false;
    globals.regions[s].initial_energy = 0.0;
    globals.regions[s].initial_density = 0.0;
    globals.regions[s].velx = 0.0;
    globals.regions[s].vely = 0.0;
  }

  glb_input.clear();
  glb_input.seekg(0);

  while (true) {
    std::getline(glb_input, line);
    if (glb_input.eof()) break;
    if (line.empty()) continue;

    std::vector<std::string> tokens;
    std::string line_copy = line;
    for (char *w = std::strtok(line_copy.data(), " ="); w != nullptr; w = std::strtok(nullptr, " =")) {
      tokens.emplace_back(w);
    }

    if (tokens[0].empty()) break;

    if (tokens[0] == "initial_timestep") {
      globals.dt_initial = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " initial_timestep " << globals.dt_initial << std::endl;
    } else if (tokens[0] == "max_timestep") {
      globals.dt_ceiling = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " max_timestep " << globals.dt_ceiling << std::endl;
    } else if (tokens[0] == "timestep_rise") {
      globals.dt_growth = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " timestep_rise " << globals.dt_growth << std::endl;
    } else if (tokens[0] == "stop_time") {
      globals.stop_time = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " stop_time " << globals.stop_time << std::endl;
    } else if (tokens[0] == "max_steps") {
      globals.max_steps = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " max_steps " << globals.max_steps << std::endl;
    } else if (tokens[0] == "x_lower") {
      globals.mesh_spec.x_lower = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " x_lower " << globals.mesh_spec.x_lower << std::endl;
    } else if (tokens[0] == "x_upper") {
      globals.mesh_spec.x_upper = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " x_upper " << globals.mesh_spec.x_upper << std::endl;
    } else if (tokens[0] == "y_lower") {
      globals.mesh_spec.y_lower = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " y_lower " << globals.mesh_spec.y_lower << std::endl;
    } else if (tokens[0] == "y_upper") {
      globals.mesh_spec.y_upper = std::stod(tokens[1]);
      if (parallel.is_root) glb_log << " y_upper " << globals.mesh_spec.y_upper << std::endl;
    } else if (tokens[0] == "grid_nx") {
      globals.mesh_spec.grid_nx = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " grid_nx " << globals.mesh_spec.grid_nx << std::endl;
    } else if (tokens[0] == "grid_ny") {
      globals.mesh_spec.grid_ny = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " grid_ny " << globals.mesh_spec.grid_ny << std::endl;
    } else if (tokens[0] == "viz_interval") {
      globals.viz_interval = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " viz_interval " << globals.viz_interval << std::endl;
    } else if (tokens[0] == "summary_interval") {
      globals.summary_interval = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " summary_interval " << globals.summary_interval << std::endl;
    } else if (tokens[0] == "tiles_per_subdomain") {
      globals.tiles_per_subdomain = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " tiles_per_subdomain " << globals.tiles_per_subdomain << std::endl;
    } else if (tokens[0] == "tiles_per_problem") {
      globals.tiles_per_subdomain = std::stoi(tokens[1]) / parallel.rank_count;
      if (parallel.is_root) glb_log << " tiles_per_subdomain " << globals.tiles_per_subdomain << std::endl;
    } else if (tokens[0] == "timing_enabled") {
      globals.timing_enabled = true;
      if (parallel.is_root) glb_log << " Profiler on" << std::endl;
    } else if (tokens[0] == "test_case_id") {
      globals.test_case_id = std::stoi(tokens[1]);
      if (parallel.is_root) glb_log << " test_case_id " << globals.test_case_id;
    } else if (tokens[0] == "region_id") {
      int region_id = std::stoi(tokens[1]) - 1;

      if (parallel.is_root) glb_log << "Reading specification for region_id " << region_id + 1 << std::endl << std::endl;
      if (globals.regions[region_id].defined) fatal_message((char *)"parse_deck", (char *)"State defined twice.");

      globals.regions[region_id].defined = true;
      for (size_t iw = 2; iw < tokens.size(); ++iw) {
        std::string w = tokens[iw];
        if (w.empty()) break;

        if (w == "velx") {
          w = tokens[++iw];
          globals.regions[region_id].velx = std::stod(w);
          if (parallel.is_root) glb_log << " velx " << globals.regions[region_id].velx << std::endl;
        } else if (w == "vely") {
          w = tokens[++iw];
          globals.regions[region_id].vely = std::stod(w);
          if (parallel.is_root) glb_log << " vely " << globals.regions[region_id].vely << std::endl;
        } else if (w == "x_lower") {
          w = tokens[++iw];
          globals.regions[region_id].x_lower = std::stod(w);
          if (parallel.is_root) glb_log << " x_lower " << globals.regions[region_id].x_lower << std::endl;
        } else if (w == "y_lower") {
          w = tokens[++iw];
          globals.regions[region_id].y_lower = std::stod(w);
          if (parallel.is_root) glb_log << " y_lower " << globals.regions[region_id].y_lower << std::endl;
        } else if (w == "x_upper") {
          w = tokens[++iw];
          globals.regions[region_id].x_upper = std::stod(w);
          if (parallel.is_root) glb_log << " x_upper " << globals.regions[region_id].x_upper << std::endl;
        } else if (w == "y_upper") {
          w = tokens[++iw];
          globals.regions[region_id].y_upper = std::stod(w);
          if (parallel.is_root) glb_log << " y_upper " << globals.regions[region_id].y_upper << std::endl;
        } else if (w == "circle_radius") {
          w = tokens[++iw];
          globals.regions[region_id].circle_radius = std::stod(w);
          if (parallel.is_root) glb_log << " circle_radius " << globals.regions[region_id].circle_radius << std::endl;
        } else if (w == "initial_density") {
          w = tokens[++iw];
          globals.regions[region_id].initial_density = std::stod(w);
          if (parallel.is_root) glb_log << " initial_density " << globals.regions[region_id].initial_density << std::endl;
        } else if (w == "initial_energy") {
          w = tokens[++iw];
          globals.regions[region_id].initial_energy = std::stod(w);
          if (parallel.is_root) glb_log << " initial_energy " << globals.regions[region_id].initial_energy << std::endl;
        } else if (w == "shape_selector") {
          w = tokens[++iw];
          if (w == "rectangle") {
            globals.regions[region_id].shape_selector = region_shape_kind::glb_rectangle;
            if (parallel.is_root) glb_log << " region_id shape_selector rectangular" << std::endl;
          } else if (w == "circle") {
            globals.regions[region_id].shape_selector = region_shape_kind::glb_circle;
            if (parallel.is_root) glb_log << " region_id shape_selector circular" << std::endl;
          } else if (w == "point") {
            globals.regions[region_id].shape_selector = region_shape_kind::glb_single_point;
            if (parallel.is_root) glb_log << " region_id shape_selector point" << std::endl;
          }
        }
      }
    }
    if (parallel.is_root) glb_log << std::endl;
  }

  if (parallel.is_root) {
    glb_log << std::endl << std::endl << "Input read finished." << std::endl << std::endl;
  }

  double dx, dy;
  dx = (globals.mesh_spec.x_upper - globals.mesh_spec.x_lower) / (float)globals.mesh_spec.grid_nx;
  dy = (globals.mesh_spec.y_upper - globals.mesh_spec.y_lower) / (float)globals.mesh_spec.grid_ny;
  for (int n = 1; n < globals.region_count; ++n) {
    globals.regions[n].x_lower += dx / 100.0;
    globals.regions[n].y_lower += dy / 100.0;
    globals.regions[n].x_upper -= dx / 100.0;
    globals.regions[n].y_upper -= dy / 100.0;
  }
}
