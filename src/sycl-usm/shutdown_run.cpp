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
    hydro2d::free(globals.context.exec_ctx, t.field.density0, t.field.density1, t.field.energy0, t.field.spec_energy1, t.field.cell_pressure,
                 t.field.shock_damping, t.field.wave_speed, t.field.velx0, t.field.velx1, t.field.vely0, t.field.vely1, t.field.volume_flux_x,
                 t.field.face_mass_flux_x, t.field.volume_flux_y, t.field.face_mass_flux_y,
                 t.field.scratch_buf1, // nodal_mass_flux, stepbymass, volume_change, start_volume
                 t.field.scratch_buf2, // nodal_mass_after, end_volume
                 t.field.scratch_buf3, // nodal_mass_before,pre_mass
                 t.field.scratch_buf4, // advec_vel, post_mass
                 t.field.scratch_buf5, // momentum_flux, advec_vol
                 t.field.scratch_buf6, // start_volume, post_ener
                 t.field.scratch_buf7, // end_volume, face_energy_flux
                 t.field.xcentre, t.field.dxcell, t.field.ycentre, t.field.dycell, t.field.xnode, t.field.dxnode, t.field.ynode,
                 t.field.dynode, t.field.cell_volume, t.field.xface_area, t.field.yface_area);
  }
}
