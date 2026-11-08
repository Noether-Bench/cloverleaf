
#pragma once

#include "device_context.h"
#include "run_config.h"

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#define glb_huge_int (640000)
#define glb_tiny (1.0e-16)
#define glb_huge (1.0e+21)
#define FIELD_COUNT (15)

#define TOKEN_JOIN(a, b) TOKEN_JOIN_A(a, b)
#define TOKEN_JOIN_A(a, b) TOKEN_JOIN_B(~, a##b)
#define TOKEN_JOIN_B(p, res) res

#define CONCAT_LINE(name_stem) TOKEN_JOIN(name_stem, __LINE__)

namespace hydro2d {
template <typename T> struct DeviceArray2D;
template <typename T> struct DeviceArray1D;

struct subdomain_context;
struct context;


}

enum field_slot {
  fld_density0 = 0,
  fld_density1 = 1,
  fld_energy0 = 2,
  fld_energy1 = 3,
  fld_pressure = 4,
  fld_viscosity = 5,
  fld_sound_speed = 6,
  fld_velx0 = 7,
  fld_velx1 = 8,
  fld_vely0 = 9,
  fld_vely1 = 10,
  fld_volume_flux_x = 11,
  fld_volume_flux_y = 12,
  fld_mass_flux_x = 13,
  fld_mass_flux_y = 14
};

enum subdomain_side_kind { subdomain_xlo = 0, subdomain_xhi = 1, subdomain_ylo = 2, subdomain_yhi = 3, mesh_edge = -1 };

enum tile_side_kind { west_tile = 0, tile_xhi = 1, tile_ylo = 3, tile_yhi = 3, tile_edge = -1 };

struct field_set {

  hydro2d::DeviceArray2D<double> density0;
  hydro2d::DeviceArray2D<double> density1;
  hydro2d::DeviceArray2D<double> energy0;
  hydro2d::DeviceArray2D<double> spec_energy1;
  hydro2d::DeviceArray2D<double> cell_pressure;
  hydro2d::DeviceArray2D<double> shock_damping;
  hydro2d::DeviceArray2D<double> wave_speed;
  hydro2d::DeviceArray2D<double> velx0, velx1;
  hydro2d::DeviceArray2D<double> vely0, vely1;
  hydro2d::DeviceArray2D<double> volume_flux_x, face_mass_flux_x;
  hydro2d::DeviceArray2D<double> volume_flux_y, face_mass_flux_y;

  hydro2d::DeviceArray2D<double> scratch_buf1;
  hydro2d::DeviceArray2D<double> scratch_buf2;
  hydro2d::DeviceArray2D<double> scratch_buf3;
  hydro2d::DeviceArray2D<double> scratch_buf4;
  hydro2d::DeviceArray2D<double> scratch_buf5;
  hydro2d::DeviceArray2D<double> scratch_buf6;
  hydro2d::DeviceArray2D<double> scratch_buf7;

  hydro2d::DeviceArray1D<double> xcentre;
  hydro2d::DeviceArray1D<double> dxcell;
  hydro2d::DeviceArray1D<double> ycentre;
  hydro2d::DeviceArray1D<double> dycell;
  hydro2d::DeviceArray1D<double> xnode;
  hydro2d::DeviceArray1D<double> dxnode;
  hydro2d::DeviceArray1D<double> ynode;
  hydro2d::DeviceArray1D<double> dynode;

  hydro2d::DeviceArray2D<double> cell_volume;
  hydro2d::DeviceArray2D<double> xface_area;
  hydro2d::DeviceArray2D<double> yface_area;

  size_t cell_row_pitch;
  size_t node_row_pitch;
  size_t xflux_row_pitch, yface_row_stride;

  explicit field_set(size_t x_extent, size_t y_extent, hydro2d::context &ctx);
};

struct tile_desc {

  std::array<int, 4> tile_adjacent;
  std::array<int, 4> edge_mask;
  int tile_xlo, blk_xmax, blk_ymin, blk_ymax;
  int tile_west_idx, right_idx, blk_bottom, blk_top;
};

struct tile_block {

  tile_desc meta;
  field_set field;

  explicit tile_block(const tile_desc &meta, hydro2d::context &ctx)
      : meta(meta),
        field((meta.blk_xmax + 2) - (meta.tile_xlo - 2) + 1, (meta.blk_ymax + 2) - (meta.blk_ymin - 2) + 1, ctx) {}
};

struct subdomain_type {

  hydro2d::subdomain_context context;


  const std::array<int, 4> subdomain_adjacent;

  const int rank_id;
  const int x_lo;
  const int y_lo;
  const int x_hi;
  const int y_hi;

  const int first_col, last_col, first_row, top;
  const int left_on_edge, right_on_edge, bottom_on_edge, top_on_edge;


  std::vector<tile_block> tile_list;

  subdomain_type(const std::array<int, 4> &subdomainAdjacent,
             int rank_id,
             int xLo, int yLo, int xHi, int yHi,
             int first_col, int last_col, int first_row, int top,
             int leftOnEdge, int rightOnEdge, int bottomOnEdge, int topOnEdge,
             int tiles_per_subdomain);
};

struct timing_table {

  double compute_dt = 0.0;
  double accel_time = 0.0;
  double PdV = 0.0;
  double cell_remap_time = 0.0;
  double momentum_remap_time = 0.0;
  double shock_damping = 0.0;
  double equation_of_state = 0.0;
  double plot_output = 0.0;
  double summary_time = 0.0;
  double reset_time = 0.0;
  double rollback_state = 0.0;
  double flux_time = 0.0;
  double tile_halo_time = 0.0;
  double local_halo_time = 0.0;
  double rank_halo_time = 0.0;
};

struct sim_state {
  const run_config settings;
  hydro2d::context context;
  subdomain_type subdomain;

  int error_status{};

  int cycle_index{};
  bool remap_in_x = true;
  double time{};

  double dt{};
  double dt_prev{};

  bool run_finished = false;
  bool test_failed = false;
  int jdt{}, kdt{};

  bool timing_enabled = false;
  timing_table kernel_timings{};

  sim_state(const run_config &settings, hydro2d::context exec_ctx, subdomain_type subdomain);
};
