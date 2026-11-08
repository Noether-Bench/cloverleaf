
#include "problem_setup.h"

#include <cmath>

static void lay_out_axis(sim_state &globals, double *node, double *node_width, double *centre, double *cell_width, size_t node_count,
                         size_t cell_count, double lower, double spacing, size_t lo) {

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int j = 0; j < (node_count); j++) {
    node[j] = lower + spacing * (j - 1 - lo);
    node_width[j] = spacing;
  }

#pragma omp target teams distribute parallel for simd clover_use_target(globals.context.use_target)
  for (int j = 0; j < (cell_count); j++) {
    centre[j] = 0.5 * (node[j] + node[j + 1]);
    cell_width[j] = spacing;
  }
}

void setup_subdomain_mesh(const int tile_id, sim_state &globals) {

  double dx = (globals.settings.mesh_spec.x_upper - globals.settings.mesh_spec.x_lower) / (double)(globals.settings.mesh_spec.grid_nx);
  double dy = (globals.settings.mesh_spec.y_upper - globals.settings.mesh_spec.y_lower) / (double)(globals.settings.mesh_spec.grid_ny);

  double x_lower = globals.settings.mesh_spec.x_lower + dx * (double)(globals.subdomain.tile_list[tile_id].meta.tile_west_idx - 1);

  double y_lower = globals.settings.mesh_spec.y_lower + dy * (double)(globals.subdomain.tile_list[tile_id].meta.blk_bottom - 1);

  size_t x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  size_t x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  size_t y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  size_t y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  size_t x_extent = (x_hi + 3) - (x_lo - 2) + 1;
  size_t y_extent = (y_hi + 3) - (y_lo - 2) + 1;

  field_set &field = globals.subdomain.tile_list[tile_id].field;

  size_t x_extent_full = (x_hi + 2) - (x_lo - 2) + 1;
  size_t y_extent_full = (y_hi + 2) - (y_lo - 2) + 1;

  lay_out_axis(globals, field.xnode.data, field.dxnode.data, field.xcentre.data, field.dxcell.data, x_extent, x_extent_full, x_lower, dx, x_lo);
  lay_out_axis(globals, field.ynode.data, field.dynode.data, field.ycentre.data, field.dycell.data, y_extent, y_extent_full, y_lower, dy, y_lo);

  double *dxcell = field.dxcell.data;
  double *dycell = field.dycell.data;

  size_t cell_row_pitch = field.cell_row_pitch;
  size_t xflux_row_pitch = field.xflux_row_pitch;
  size_t yface_row_stride = field.yface_row_stride;

  double *cell_volume = field.cell_volume.data;
  double *xface_area = field.xface_area.data;
  double *yface_area = field.yface_area.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
  for (int j = 0; j < (y_extent_full); j++) {
    for (int i = 0; i < (x_extent_full); i++) {
      cell_volume[i + j * cell_row_pitch] = dx * dy;
      xface_area[i + j * xflux_row_pitch] = dycell[j];
      yface_area[i + j * yface_row_stride] = dxcell[i];
    }
  }
}

void populate_subdomain(const int tile_id, sim_state &globals) {

  hydro2d::DeviceArray1D<double> region_density_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_energy_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_velx_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_vely_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_xlo_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_xhi_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_ylo_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_yhi_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<double> region_radius_buf(globals.context, globals.settings.region_count);
  hydro2d::DeviceArray1D<int> region_shape_buf(globals.context, globals.settings.region_count);

  for (int region_id = 0; region_id < globals.settings.region_count; ++region_id) {
    region_density_buf[region_id] = globals.settings.regions[region_id].initial_density;
    region_energy_buf[region_id] = globals.settings.regions[region_id].initial_energy;
    region_velx_buf[region_id] = globals.settings.regions[region_id].velx;
    region_vely_buf[region_id] = globals.settings.regions[region_id].vely;
    region_xlo_buf[region_id] = globals.settings.regions[region_id].x_lower;
    region_xhi_buf[region_id] = globals.settings.regions[region_id].x_upper;
    region_ylo_buf[region_id] = globals.settings.regions[region_id].y_lower;
    region_yhi_buf[region_id] = globals.settings.regions[region_id].y_upper;
    region_radius_buf[region_id] = globals.settings.regions[region_id].circle_radius;
    region_shape_buf[region_id] = globals.settings.regions[region_id].shape_selector;
  }


  const int x_lo = globals.subdomain.tile_list[tile_id].meta.tile_xlo;
  const int x_hi = globals.subdomain.tile_list[tile_id].meta.blk_xmax;
  const int y_lo = globals.subdomain.tile_list[tile_id].meta.blk_ymin;
  const int y_hi = globals.subdomain.tile_list[tile_id].meta.blk_ymax;

  int x_extent = (x_hi + 2) - (x_lo - 2) + 1;
  int y_extent = (y_hi + 2) - (y_lo - 2) + 1;


  field_set &field = globals.subdomain.tile_list[tile_id].field;

  const double background_energy = region_energy_buf[0];
  const double background_density = region_density_buf[0];
  const double background_velx = region_velx_buf[0];
  const double background_vely = region_vely_buf[0];

  const int cell_row_pitch = field.cell_row_pitch;
  const int node_row_pitch = field.node_row_pitch;

  double *energy0 = field.energy0.data;
  double *density0 = field.density0.data;
  double *velx0 = field.velx0.data;
  double *vely0 = field.vely0.data;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)
  for (int j = 0; j < (y_extent); j++) {
    for (int i = 0; i < (x_extent); i++) {
      energy0[i + j * cell_row_pitch] = background_energy;
      density0[i + j * cell_row_pitch] = background_density;
      velx0[i + j * node_row_pitch] = background_velx;
      vely0[i + j * node_row_pitch] = background_vely;
    }
  }

  double *xcentre = field.xcentre.data;
  double *ycentre = field.ycentre.data;

  double *xnode = field.xnode.data;
  double *ynode = field.ynode.data;

  const double *region_density = region_density_buf.data;
  const double *region_energy = region_energy_buf.data;
  const double *region_velx = region_velx_buf.data;
  const double *region_vely = region_vely_buf.data;
  const double *region_xlo = region_xlo_buf.data;
  const double *region_xhi = region_xhi_buf.data;
  const double *region_ylo = region_ylo_buf.data;
  const double *region_yhi = region_yhi_buf.data;
  const double *region_radius = region_radius_buf.data;
  const int *region_shape_code = region_shape_buf.data;

  for (int region_id = 1; region_id < globals.settings.region_count; ++region_id) {
#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(globals.context.use_target)                            \
    map(to : region_density[ : region_density_buf.N()]) map(to : region_energy[ : region_energy_buf.N()])                                \
    map(to : region_velx[ : region_velx_buf.N()]) map(to : region_vely[ : region_vely_buf.N()])                                          \
    map(to : region_xlo[ : region_xlo_buf.N()]) map(to : region_xhi[ : region_xhi_buf.N()])                                          \
    map(to : region_ylo[ : region_ylo_buf.N()]) map(to : region_yhi[ : region_yhi_buf.N()])                                          \
    map(to : region_radius[ : region_radius_buf.N()]) map(to : region_shape_code[ : region_shape_buf.N()])
    for (int j = 0; j < (y_extent); j++) {
      for (int i = 0; i < (x_extent); i++) {
        double centre_x = region_xlo[region_id];
        double centre_y = region_ylo[region_id];
        if (region_shape_code[region_id] == glb_rectangle) {
          if (xnode[i + 1] >= region_xlo[region_id] && xnode[i] < region_xhi[region_id]) {
            if (ynode[j + 1] >= region_ylo[region_id] && ynode[j] < region_yhi[region_id]) {
              energy0[i + j * cell_row_pitch] = region_energy[region_id];
              density0[i + j * cell_row_pitch] = region_density[region_id];
              for (int kt = j; kt <= j + 1; ++kt) {
                for (int jt = i; jt <= i + 1; ++jt) {
                  velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                  vely0[jt + kt * node_row_pitch] = region_vely[region_id];
                }
              }
            }
          }
        } else if (region_shape_code[region_id] == glb_circle) {
          double circle_radius = sqrt((xcentre[i] - centre_x) * (xcentre[i] - centre_x) + (ycentre[j] - centre_y) * (ycentre[j] - centre_y));
          if (circle_radius <= region_radius[region_id]) {
            energy0[i + j * cell_row_pitch] = region_energy[region_id];
            density0[i + j * cell_row_pitch] = region_density[region_id];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                vely0[jt + kt * node_row_pitch] = region_vely[region_id];
              }
            }
          }
        } else if (region_shape_code[region_id] == glb_single_point) {
          if (xnode[i] == centre_x && ynode[j] == centre_y) {
            energy0[i + j * cell_row_pitch] = region_energy[region_id];
            density0[i + j * cell_row_pitch] = region_density[region_id];
            for (int kt = j; kt <= j + 1; ++kt) {
              for (int jt = i; jt <= i + 1; ++jt) {
                velx0[jt + kt * node_row_pitch] = region_velx[region_id];
                vely0[jt + kt * node_row_pitch] = region_vely[region_id];
              }
            }
          }
        }
      }
    }
  }
}
