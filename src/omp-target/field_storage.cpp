
#include "field_storage.h"

static void clear_coordinate_pair(bool use_target, double *position, double *spacing, int length) {
#pragma omp target teams distribute parallel for simd clover_use_target(use_target)
  for (int id = 0; id < (length); id++) {
    position[id] = 0.0;
    spacing[id] = 0.0;
  }
}

void allocate_fields(sim_state &globals) {

  const bool use_target = globals.context.use_target;

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    field_set &field = t.field;

    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *wave_speed = field.wave_speed.data;
    double *vely0 = field.vely0.data;
    double *vely1 = field.vely1.data;
    double *velx0 = field.velx0.data;
    double *velx1 = field.velx1.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *volume_flux_y = field.volume_flux_y.data;
    double *face_mass_flux_x = field.face_mass_flux_x.data;
    double *face_mass_flux_y = field.face_mass_flux_y.data;
    double *scratch_buf1 = field.scratch_buf1.data;
    double *scratch_buf2 = field.scratch_buf2.data;
    double *scratch_buf3 = field.scratch_buf3.data;
    double *scratch_buf4 = field.scratch_buf4.data;
    double *scratch_buf5 = field.scratch_buf5.data;
    double *scratch_buf6 = field.scratch_buf6.data;
    double *scratch_buf7 = field.scratch_buf7.data;
    double *xcentre = field.xcentre.data;
    double *dxcell = field.dxcell.data;
    double *ycentre = field.ycentre.data;
    double *dycell = field.dycell.data;
    double *xnode = field.xnode.data;
    double *dxnode = field.dxnode.data;
    double *ynode = field.ynode.data;
    double *dynode = field.dynode.data;
    double *cell_volume = field.cell_volume.data;
    double *xface_area = field.xface_area.data;
    double *yface_area = field.yface_area.data;

#pragma omp target enter data map(alloc : density0[ : field.density0.N()]) map(alloc : density1[ : field.density1.N()])                    \
    map(alloc : energy0[ : field.energy0.N()]) map(alloc : spec_energy1[ : field.spec_energy1.N()]) map(alloc : cell_pressure[ : field.cell_pressure.N()])     \
    map(alloc : shock_damping[ : field.shock_damping.N()]) map(alloc : wave_speed[ : field.wave_speed.N()]) map(alloc : vely0[ : field.vely0.N()]) \
    map(alloc : vely1[ : field.vely1.N()]) map(alloc : velx0[ : field.velx0.N()]) map(alloc : velx1[ : field.velx1.N()])                   \
    map(alloc : volume_flux_x[ : field.volume_flux_x.N()]) map(alloc : volume_flux_y[ : field.volume_flux_y.N()])                                      \
    map(alloc : face_mass_flux_x[ : field.face_mass_flux_x.N()]) map(alloc : face_mass_flux_y[ : field.face_mass_flux_y.N()])                                  \
    map(alloc : scratch_buf1[ : field.scratch_buf1.N()]) map(alloc : scratch_buf2[ : field.scratch_buf2.N()])                                  \
    map(alloc : scratch_buf3[ : field.scratch_buf3.N()]) map(alloc : scratch_buf4[ : field.scratch_buf4.N()])                                  \
    map(alloc : scratch_buf5[ : field.scratch_buf5.N()]) map(alloc : scratch_buf6[ : field.scratch_buf6.N()])                                  \
    map(alloc : scratch_buf7[ : field.scratch_buf7.N()]) map(alloc : xcentre[ : field.xcentre.N()]) map(alloc : dxcell[ : field.dxcell.N()])     \
    map(alloc : ycentre[ : field.ycentre.N()]) map(alloc : dycell[ : field.dycell.N()]) map(alloc : xnode[ : field.xnode.N()])             \
    map(alloc : dxnode[ : field.dxnode.N()]) map(alloc : ynode[ : field.ynode.N()]) map(alloc : dynode[ : field.dynode.N()])   \
    map(alloc : cell_volume[ : field.cell_volume.N()]) map(alloc : xface_area[ : field.xface_area.N()]) map(alloc : yface_area[ : field.yface_area.N()])

    const int x_extent = (t.meta.blk_xmax + 2) - (t.meta.tile_xlo - 2) + 1;
    const int y_extent = (t.meta.blk_ymax + 2) - (t.meta.blk_ymin - 2) + 1;
    const int node_x_extent = x_extent + 1;
    const int node_y_extent = y_extent + 1;

    const int node_row_pitch = field.node_row_pitch;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = 0; j < node_y_extent; j++) {
      for (int i = 0; i < node_x_extent; i++) {
        scratch_buf1[i + j * node_row_pitch] = 0.0;
        scratch_buf2[i + j * node_row_pitch] = 0.0;
        scratch_buf3[i + j * node_row_pitch] = 0.0;
        scratch_buf4[i + j * node_row_pitch] = 0.0;
        scratch_buf5[i + j * node_row_pitch] = 0.0;
        scratch_buf6[i + j * node_row_pitch] = 0.0;
        scratch_buf7[i + j * node_row_pitch] = 0.0;
        velx0[i + j * node_row_pitch] = 0.0;
        velx1[i + j * node_row_pitch] = 0.0;
        vely0[i + j * node_row_pitch] = 0.0;
        vely1[i + j * node_row_pitch] = 0.0;
      }
    }

    const int cell_row_pitch = field.cell_row_pitch;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = 0; j < (y_extent); j++) {
      for (int i = 0; i < (x_extent); i++) {
        density0[i + j * cell_row_pitch] = 0.0;
        density1[i + j * cell_row_pitch] = 0.0;
        energy0[i + j * cell_row_pitch] = 0.0;
        spec_energy1[i + j * cell_row_pitch] = 0.0;
        cell_pressure[i + j * cell_row_pitch] = 0.0;
        shock_damping[i + j * cell_row_pitch] = 0.0;
        wave_speed[i + j * cell_row_pitch] = 0.0;
        cell_volume[i + j * cell_row_pitch] = 0.0;
      }
    }

    const int xflux_row_pitch = field.xflux_row_pitch;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = 0; j < (y_extent); j++) {
      for (int i = 0; i < (x_extent); i++) {
        volume_flux_x[i + j * xflux_row_pitch] = 0.0;
        face_mass_flux_x[i + j * xflux_row_pitch] = 0.0;
        xface_area[i + j * xflux_row_pitch] = 0.0;
      }
    }

    const int yface_row_stride = field.yface_row_stride;

#pragma omp target teams distribute parallel for simd collapse(2) clover_use_target(use_target)
    for (int j = 0; j < node_y_extent; j++) {
      for (int i = 0; i < (x_extent); i++) {
        volume_flux_y[i + j * yface_row_stride] = 0.0;
        face_mass_flux_y[i + j * yface_row_stride] = 0.0;
        yface_area[i + j * yface_row_stride] = 0.0;
      }
    }

    clear_coordinate_pair(use_target, xcentre, dxcell, x_extent);
    clear_coordinate_pair(use_target, ycentre, dycell, y_extent);
    clear_coordinate_pair(use_target, xnode, dxnode, node_x_extent);
    clear_coordinate_pair(use_target, ynode, dynode, node_y_extent);
  }
}

void hydro2d_alloc_comm_buffers(sim_state &globals, mpi_context &parallel) {

  if (parallel.rank_id == globals.subdomain.rank_id) {

  }
}

void shutdown_run(sim_state &globals) {

  for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {

    tile_block &t = globals.subdomain.tile_list[tile_id];
    field_set &field = t.field;

    double *density0 = field.density0.data;
    double *density1 = field.density1.data;
    double *energy0 = field.energy0.data;
    double *spec_energy1 = field.spec_energy1.data;
    double *cell_pressure = field.cell_pressure.data;
    double *shock_damping = field.shock_damping.data;
    double *wave_speed = field.wave_speed.data;
    double *vely0 = field.vely0.data;
    double *vely1 = field.vely1.data;
    double *velx0 = field.velx0.data;
    double *velx1 = field.velx1.data;
    double *volume_flux_x = field.volume_flux_x.data;
    double *volume_flux_y = field.volume_flux_y.data;
    double *face_mass_flux_x = field.face_mass_flux_x.data;
    double *face_mass_flux_y = field.face_mass_flux_y.data;
    double *scratch_buf1 = field.scratch_buf1.data;
    double *scratch_buf2 = field.scratch_buf2.data;
    double *scratch_buf3 = field.scratch_buf3.data;
    double *scratch_buf4 = field.scratch_buf4.data;
    double *scratch_buf5 = field.scratch_buf5.data;
    double *scratch_buf6 = field.scratch_buf6.data;
    double *scratch_buf7 = field.scratch_buf7.data;
    double *xcentre = field.xcentre.data;
    double *dxcell = field.dxcell.data;
    double *ycentre = field.ycentre.data;
    double *dycell = field.dycell.data;
    double *xnode = field.xnode.data;
    double *dxnode = field.dxnode.data;
    double *ynode = field.ynode.data;
    double *dynode = field.dynode.data;
    double *cell_volume = field.cell_volume.data;
    double *xface_area = field.xface_area.data;
    double *yface_area = field.yface_area.data;

#pragma omp target exit data map(release : density0[ : 0]) map(release : density1[ : 0]) map(release : energy0[ : 0])                      \
    map(release : spec_energy1[ : 0]) map(release : cell_pressure[ : 0]) map(release : shock_damping[ : 0]) map(release : wave_speed[ : 0])              \
    map(release : vely0[ : 0]) map(release : vely1[ : 0]) map(release : velx0[ : 0]) map(release : velx1[ : 0])                            \
    map(release : volume_flux_x[ : 0]) map(release : volume_flux_y[ : 0]) map(release : face_mass_flux_x[ : 0]) map(release : face_mass_flux_y[ : 0])      \
    map(release : scratch_buf1[ : 0]) map(release : scratch_buf2[ : 0]) map(release : scratch_buf3[ : 0]) map(release : scratch_buf4[ : 0])    \
    map(release : scratch_buf5[ : 0]) map(release : scratch_buf6[ : 0]) map(release : scratch_buf7[ : 0]) map(release : xcentre[ : 0])          \
    map(release : dxcell[ : 0]) map(release : ycentre[ : 0]) map(release : dycell[ : 0]) map(release : xnode[ : 0])                        \
    map(release : dxnode[ : 0]) map(release : ynode[ : 0]) map(release : dynode[ : 0]) map(release : cell_volume[ : 0])                   \
    map(release : xface_area[ : 0]) map(release : yface_area[ : 0])
  }
}
