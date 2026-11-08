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
  for (tile_block &t : globals.subdomain.tile_list) {
    t.field.density0.release();
    t.field.density1.release();
    t.field.energy0.release();
    t.field.spec_energy1.release();
    t.field.cell_pressure.release();
    t.field.shock_damping.release();
    t.field.wave_speed.release();
    t.field.velx0.release();
    t.field.velx1.release();
    t.field.vely0.release();
    t.field.vely1.release();
    t.field.volume_flux_x.release();
    t.field.face_mass_flux_x.release();
    t.field.volume_flux_y.release();
    t.field.face_mass_flux_y.release();
    t.field.scratch_buf1.release();
    t.field.scratch_buf2.release();
    t.field.scratch_buf3.release();
    t.field.scratch_buf4.release();
    t.field.scratch_buf5.release();
    t.field.scratch_buf6.release();
    t.field.scratch_buf7.release();
    t.field.xcentre.release();
    t.field.dxcell.release();
    t.field.ycentre.release();
    t.field.dycell.release();
    t.field.xnode.release();
    t.field.dxnode.release();
    t.field.ynode.release();

    t.field.dynode.release();
    t.field.cell_volume.release();
    t.field.xface_area.release();
    t.field.yface_area.release();
  }
}
