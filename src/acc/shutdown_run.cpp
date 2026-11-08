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

#include "shutdown_run.h"

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

#pragma acc exit data delete(density0, density1, energy0, spec_energy1, cell_pressure, shock_damping, \
	wave_speed, vely0, vely1, velx0, velx1, volume_flux_x, volume_flux_y, face_mass_flux_x, face_mass_flux_y, \
    scratch_buf1, scratch_buf2, scratch_buf3, scratch_buf4, scratch_buf5, scratch_buf6, scratch_buf7, \
    xcentre, dxcell, ycentre, dycell, xnode, dxnode, ynode, dynode, cell_volume, xface_area, yface_area)
  }
}
