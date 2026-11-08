
#include "field_output.h"
#include "cell_state.h"
#include "walltime.h"
#include "ghost_cells.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>


static bool is_first_dump = true;

void plot_output(sim_state &globals, mpi_context &parallel) {

  std::string label = "hydro2d";
  if (parallel.is_root) {

    if (is_first_dump) {

      int total_tile_count = globals.settings.total_subdomains * globals.settings.tiles_per_subdomain;
      std::string out_path = "hydro2d.plot_output";
      std::ofstream u;
      u.open(out_path);
      u << "!NBLOCKS " << total_tile_count << std::endl;
      u.close();
      is_first_dump = false;
    }
  }

  double phase_start{};
  if (globals.timing_enabled) phase_start = wall_seconds();
  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    equation_of_state(globals, tile_id, false);
  }
  if (globals.timing_enabled) globals.kernel_timings.equation_of_state += wall_seconds() - phase_start;

  int exchange_mask[FIELD_COUNT];
  for (int i = 0; i < FIELD_COUNT; ++i)
    exchange_mask[i] = 0;
  exchange_mask[fld_pressure] = 1;
  exchange_mask[fld_velx0] = 1;
  exchange_mask[fld_vely0] = 1;
  refresh_ghost_cells(globals, exchange_mask, 1);

  if (globals.timing_enabled) phase_start = wall_seconds();
  shock_damping(globals);
  if (globals.timing_enabled) globals.kernel_timings.shock_damping += wall_seconds() - phase_start;

  if (parallel.is_root) {

    std::string out_path = "hydro2d.plot_output";
    std::ofstream u;
    u.open(out_path, std::ios::app);
    for (int c = 0; c < parallel.rank_count; ++c) {
      std::stringstream file_name_builder;
      file_name_builder << "." << std::setfill('0') << std::setw(5) << c;
      for (int tile_id = 1; tile_id <= globals.settings.tiles_per_subdomain; ++tile_id) {
        file_name_builder << "." << std::setfill('0') << std::setw(5) << tile_id;
        file_name_builder << "." << std::setfill('0') << std::setw(5) << globals.cycle_index;
        file_name_builder << ".vtk";
        u << label << file_name_builder.str() << std::endl;
      }
    }
    u.close();
  }

  if (globals.timing_enabled) phase_start = wall_seconds();

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
    if (globals.subdomain.rank_id == parallel.rank_id) {
      int nxc = globals.subdomain.tile_list[tile_id].meta.blk_xmax - globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1;
      int nyc = globals.subdomain.tile_list[tile_id].meta.blk_ymax - globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1;
      int nxv = nxc + 1;
      int nyv = nyc + 1;

      std::stringstream file_name_builder;
      file_name_builder << label;
      file_name_builder << "." << std::setfill('0') << std::setw(5) << parallel.rank_id;
      file_name_builder << "." << std::setfill('0') << std::setw(5) << tile_id + 1;
      file_name_builder << "." << std::setfill('0') << std::setw(5) << globals.cycle_index;
      file_name_builder << ".vtk";
      std::ofstream u;
      u.open(file_name_builder.str());
      u << "# vtk DataFile Version 3.0" << std::endl;
      u << "vtk output" << std::endl;
      u << "ASCII" << std::endl;
      u << "DATASET RECTILINEAR_GRID" << std::endl;
      u << "DIMENSIONS " << nxv << " " << nyv << " 1" << std::endl;
      u << "X_COORDINATES " << nxv << " double" << std::endl;

      auto host_xnode = globals.subdomain.tile_list[tile_id].field.xnode.host_copy();

      for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1 + 1; ++j) {
        u << host_xnode[j] << std::endl;
      }

      u << "Y_COORDINATES " << nyv << " double" << std::endl;

      auto host_ynode = globals.subdomain.tile_list[tile_id].field.ynode.host_copy();

      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1 + 1; ++k) {
        u << host_ynode[k] << std::endl;
      }

      u << "Z_COORDINATES 1 double" << std::endl;
      u << "0" << std::endl;

      u << "CELL_DATA " << nxc * nyc << std::endl;
      u << "FIELD FieldData 4" << std::endl;
      u << "initial_density 1 " << nxc * nyc << " double" << std::endl;
      auto host_density0 = globals.subdomain.tile_list[tile_id].field.density0.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << host_density0(j, k) << std::endl;
        }
      }

      u << "initial_energy 1 " << nxc * nyc << " double" << std::endl;
      auto host_energy0 = globals.subdomain.tile_list[tile_id].field.energy0.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << host_energy0(j, k) << std::endl;
        }
      }

      u << "cell_pressure 1 " << nxc * nyc << " double" << std::endl;
      auto host_pressure = globals.subdomain.tile_list[tile_id].field.cell_pressure.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1; ++j) {
          u << std::scientific << std::setprecision(3) << host_pressure(j, k) << std::endl;
        }
      }

      u << "shock_damping 1 " << nxc * nyc << " double" << std::endl;
      auto host_viscosity = globals.subdomain.tile_list[tile_id].field.shock_damping.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1; ++j) {
          double clipped_value = (std::fabs(host_viscosity(j, k)) > 0.00000001) ? host_viscosity(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << clipped_value << std::endl;
        }
      }

      u << "POINT_DATA " << nxv * nyv << std::endl;
      u << "FIELD FieldData 2" << std::endl;
      u << "x_vel 1 " << nxv * nyv << " double" << std::endl;
      auto host_velx0 = globals.subdomain.tile_list[tile_id].field.velx0.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1 + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1 + 1; ++j) {
          double clipped_value = (std::fabs(host_velx0(j, k)) > 0.00000001) ? host_velx0(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << clipped_value << std::endl;
        }
      }
      u << "y_vel 1 " << nxv * nyv << " double" << std::endl;
      auto host_vely0 = globals.subdomain.tile_list[tile_id].field.vely0.host_copy2d();
      for (int k = globals.subdomain.tile_list[tile_id].meta.blk_ymin + 1; k <= globals.subdomain.tile_list[tile_id].meta.blk_ymax + 1 + 1; ++k) {
        for (int j = globals.subdomain.tile_list[tile_id].meta.tile_xlo + 1; j <= globals.subdomain.tile_list[tile_id].meta.blk_xmax + 1 + 1; ++j) {
          double clipped_value = (std::fabs(host_vely0(j, k)) > 0.00000001) ? host_vely0(j, k) : 0.0;
          u << std::scientific << std::setprecision(3) << clipped_value << std::endl;
        }
      }
      u.close();
    }
  }
  if (globals.timing_enabled) globals.kernel_timings.plot_output += wall_seconds() - phase_start;
}
