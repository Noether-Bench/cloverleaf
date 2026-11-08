

#include <cmath>
#include <iomanip>
#include <iostream>

#include "rank_comm.h"
#include "run_log.h"

void fatal_message(char *origin_site, char *error) {

  std::cout << std::endl
            << " Error from " << origin_site << ":" << std::endl
            << error << std::endl
            << " HYDRO2D is terminating." << std::endl
            << std::endl;

  glb_log << std::endl
        << "Error from " << origin_site << ":" << std::endl
        << error << std::endl
        << "HYDRO2D is terminating." << std::endl
        << std::endl;

  hydro2d_terminate();
}

void hydro2d_print_cycle_banner(sim_state &globals, mpi_context &parallel) {
  if (parallel.is_root) {
    glb_log << std::endl
          << "Time " << globals.time << std::endl
          << "                "
          << "Volume          "
          << "Mass            "
          << "Density         "
          << "Pressure        "
          << "Internal Energy "
          << "Kinetic Energy  "
          << "Total Energy    " << std::endl;
  }
}

void hydro2d_print_cycle(sim_state &globals, mpi_context &parallel,
                        double vol, double mass, double ie, double ke, double press) {
  if (parallel.is_root) {
    auto saved_flags = glb_log.flags();
    glb_log << " cycle_index: " << globals.cycle_index << std::scientific << std::setw(15) << vol << std::scientific << std::setw(15) << mass
          << std::scientific << std::setw(15) << mass / vol << std::scientific << std::setw(15) << press / vol << std::scientific
          << std::setw(15) << ie << std::scientific << std::setw(15) << ke << std::scientific << std::setw(15) << ie + ke << std::endl
          << std::endl;
    glb_log.flags(saved_flags);
  }
}
