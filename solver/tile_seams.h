
#pragma once

#include "device_context.h"
#include "sim_state.h"

void copy_tile_ghosts_west_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
                               hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
                               hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
                               hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1,
                               hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y,
                               hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int west_xlo, int west_xhi,
                               int west_ylo, int west_yhi, hydro2d::DeviceArray2D<double> &west_density0,
                               hydro2d::DeviceArray2D<double> &west_energy0, hydro2d::DeviceArray2D<double> &west_pressure,
                               hydro2d::DeviceArray2D<double> &west_viscosity, hydro2d::DeviceArray2D<double> &west_sound_speed,
                               hydro2d::DeviceArray2D<double> &west_density1, hydro2d::DeviceArray2D<double> &west_energy1,
                               hydro2d::DeviceArray2D<double> &west_velx0, hydro2d::DeviceArray2D<double> &west_vely0,
                               hydro2d::DeviceArray2D<double> &west_velx1, hydro2d::DeviceArray2D<double> &west_vely1,
                               hydro2d::DeviceArray2D<double> &west_volume_flux_x, hydro2d::DeviceArray2D<double> &west_volume_flux_y,
                               hydro2d::DeviceArray2D<double> &west_mass_flux_x, hydro2d::DeviceArray2D<double> &west_mass_flux_y,
                               const int exchange_mask[FIELD_COUNT], int halo_width);

void copy_tile_ghosts_east_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
                               hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
                               hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
                               hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1,
                               hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y,
                               hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int east_xlo, int east_xhi,
                               int east_ylo, int east_yhi, hydro2d::DeviceArray2D<double> &east_density0,
                               hydro2d::DeviceArray2D<double> &east_energy0, hydro2d::DeviceArray2D<double> &east_pressure,
                               hydro2d::DeviceArray2D<double> &east_viscosity, hydro2d::DeviceArray2D<double> &east_sound_speed,
                               hydro2d::DeviceArray2D<double> &east_density1, hydro2d::DeviceArray2D<double> &east_energy1,
                               hydro2d::DeviceArray2D<double> &east_velx0, hydro2d::DeviceArray2D<double> &east_vely0,
                               hydro2d::DeviceArray2D<double> &east_velx1, hydro2d::DeviceArray2D<double> &east_vely1,
                               hydro2d::DeviceArray2D<double> &east_volume_flux_x, hydro2d::DeviceArray2D<double> &east_volume_flux_y,
                               hydro2d::DeviceArray2D<double> &east_mass_flux_x, hydro2d::DeviceArray2D<double> &east_mass_flux_y,
                               const int exchange_mask[FIELD_COUNT], int halo_width);

void copy_tile_ghosts_south_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
                               hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
                               hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
                               hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1,
                               hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y,
                               hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int south_xlo,
                               int south_xhi, int south_ylo, int south_yhi, hydro2d::DeviceArray2D<double> &south_density0,
                               hydro2d::DeviceArray2D<double> &south_energy0, hydro2d::DeviceArray2D<double> &south_pressure,
                               hydro2d::DeviceArray2D<double> &south_viscosity, hydro2d::DeviceArray2D<double> &south_sound_speed,
                               hydro2d::DeviceArray2D<double> &south_density1, hydro2d::DeviceArray2D<double> &south_energy1,
                               hydro2d::DeviceArray2D<double> &south_velx0, hydro2d::DeviceArray2D<double> &south_vely0,
                               hydro2d::DeviceArray2D<double> &south_velx1, hydro2d::DeviceArray2D<double> &south_vely1,
                               hydro2d::DeviceArray2D<double> &south_volume_flux_x, hydro2d::DeviceArray2D<double> &south_volume_flux_y,
                               hydro2d::DeviceArray2D<double> &south_mass_flux_x, hydro2d::DeviceArray2D<double> &south_mass_flux_y,
                               const int exchange_mask[FIELD_COUNT], int halo_width);

void copy_tile_ghosts_north_kernel(sim_state &globals, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &density0,
                               hydro2d::DeviceArray2D<double> &energy0, hydro2d::DeviceArray2D<double> &cell_pressure, hydro2d::DeviceArray2D<double> &shock_damping,
                               hydro2d::DeviceArray2D<double> &wave_speed, hydro2d::DeviceArray2D<double> &density1, hydro2d::DeviceArray2D<double> &spec_energy1,
                               hydro2d::DeviceArray2D<double> &velx0, hydro2d::DeviceArray2D<double> &vely0, hydro2d::DeviceArray2D<double> &velx1,
                               hydro2d::DeviceArray2D<double> &vely1, hydro2d::DeviceArray2D<double> &volume_flux_x, hydro2d::DeviceArray2D<double> &volume_flux_y,
                               hydro2d::DeviceArray2D<double> &face_mass_flux_x, hydro2d::DeviceArray2D<double> &face_mass_flux_y, int north_xlo, int north_xhi,
                               int north_ylo, int north_yhi, hydro2d::DeviceArray2D<double> &north_density0, hydro2d::DeviceArray2D<double> &north_energy0,
                               hydro2d::DeviceArray2D<double> &north_pressure, hydro2d::DeviceArray2D<double> &north_viscosity,
                               hydro2d::DeviceArray2D<double> &north_sound_speed, hydro2d::DeviceArray2D<double> &north_density1,
                               hydro2d::DeviceArray2D<double> &north_energy1, hydro2d::DeviceArray2D<double> &north_velx0,
                               hydro2d::DeviceArray2D<double> &north_vely0, hydro2d::DeviceArray2D<double> &north_velx1,
                               hydro2d::DeviceArray2D<double> &north_vely1, hydro2d::DeviceArray2D<double> &north_volume_flux_x,
                               hydro2d::DeviceArray2D<double> &north_volume_flux_y, hydro2d::DeviceArray2D<double> &north_mass_flux_x,
                               hydro2d::DeviceArray2D<double> &north_mass_flux_y, const int exchange_mask[FIELD_COUNT], int halo_width);
