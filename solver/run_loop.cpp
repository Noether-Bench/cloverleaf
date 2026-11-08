
#include "run_loop.h"
#include "lagrangian.h"
#include "remap_sweeps.h"
#include "field_output.h"
#include "field_dump.h"
#include "walltime.h"
#include "stable_dt.h"

extern std::ostream glb_log;

void time_march(sim_state &globals, mpi_context &parallel) {

  double run_start = wall_seconds();

  if (!globals.settings.dumpPath.empty())
    hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_05_run_loop.txt");

  while (true) {

    double step_start = wall_seconds();

    globals.cycle_index += 1;

    compute_dt(globals, parallel);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_1_cfl_limits.txt");

    PdV(globals, true);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_2_cell_compression_half.txt");

    advance_velocity(globals);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_3_nodal_motion.txt");

    PdV(globals, false);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_4_cell_compression_full.txt");

    compute_face_fluxes(globals);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_5_face_fluxes.txt");

    flow_sweep_driver(globals);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_6_remap_sweeps.txt");

    roll_time_level(globals);
    if (!globals.settings.dumpPath.empty())
      hydro2d::write_snapshot(globals, std::to_string(parallel.rank_id) + "_" + std::to_string(globals.cycle_index) + "_7_time_levels.txt");

    globals.remap_in_x = !globals.remap_in_x;

    globals.time += globals.dt;

    if (globals.settings.summary_interval != 0 && globals.cycle_index % globals.settings.summary_interval == 0)
      report_field_totals(globals, parallel);
    if (globals.settings.viz_interval != 0 && globals.cycle_index % globals.settings.viz_interval == 0)
      plot_output(globals, parallel);

    double elapsed_wall{};
    double step1_secs = 0, step2_secs = 0;
    if (globals.cycle_index == 1) step1_secs = wall_seconds() - step_start;
    if (globals.cycle_index == 2) step2_secs = wall_seconds() - step_start;

    if (globals.time + glb_tiny > globals.settings.stop_time || globals.cycle_index >= globals.settings.max_steps) {

      globals.run_finished = true;
      report_field_totals(globals, parallel);
      if (globals.settings.viz_interval != 0) plot_output(globals, parallel);

      elapsed_wall = wall_seconds() - run_start;
      if (parallel.is_root) {
        glb_log << std::endl
              << "Calculation run_finished" << std::endl
              << "Hydro2d is finishing" << std::endl
              << "Wall clock " << elapsed_wall << std::endl
              << "First cycle_index overhead " << step1_secs - step2_secs << std::endl;
        std::cout << " Wall clock " << elapsed_wall << std::endl
                  << " First cycle_index overhead " << step1_secs - step2_secs << std::endl;
      }

      std::vector<double> rank_values(parallel.rank_count);
      if (globals.timing_enabled) {
        timing_table &p = globals.kernel_timings;
        double kernels_sum = p.compute_dt + p.equation_of_state + p.shock_damping + p.PdV + p.rollback_state + p.accel_time + p.flux_time + p.cell_remap_time +
                              p.momentum_remap_time + p.reset_time + p.summary_time + p.plot_output + p.tile_halo_time + p.local_halo_time +
                              p.rank_halo_time;
        hydro2d_gather_all(kernels_sum, rank_values);

        int loc = argmax_index(rank_values, parallel.rank_count);
        kernels_sum = rank_values[loc];

        hydro2d_gather_all(p.compute_dt, rank_values);
        p.compute_dt = rank_values[loc];
        hydro2d_gather_all(p.equation_of_state, rank_values);
        p.equation_of_state = rank_values[loc];
        hydro2d_gather_all(p.shock_damping, rank_values);
        p.shock_damping = rank_values[loc];
        hydro2d_gather_all(p.PdV, rank_values);
        p.PdV = rank_values[loc];
        hydro2d_gather_all(p.rollback_state, rank_values);
        p.rollback_state = rank_values[loc];
        hydro2d_gather_all(p.accel_time, rank_values);
        p.accel_time = rank_values[loc];
        hydro2d_gather_all(p.flux_time, rank_values);
        p.flux_time = rank_values[loc];
        hydro2d_gather_all(p.cell_remap_time, rank_values);
        p.cell_remap_time = rank_values[loc];
        hydro2d_gather_all(p.momentum_remap_time, rank_values);
        p.momentum_remap_time = rank_values[loc];
        hydro2d_gather_all(p.reset_time, rank_values);
        p.reset_time = rank_values[loc];
        hydro2d_gather_all(p.tile_halo_time, rank_values);
        p.tile_halo_time = rank_values[loc];
        hydro2d_gather_all(p.local_halo_time, rank_values);
        p.local_halo_time = rank_values[loc];
        hydro2d_gather_all(p.rank_halo_time, rank_values);
        p.rank_halo_time = rank_values[loc];
        hydro2d_gather_all(p.summary_time, rank_values);
        p.summary_time = rank_values[loc];
        hydro2d_gather_all(p.plot_output, rank_values);
        p.plot_output = rank_values[loc];

        if (parallel.is_root) {
          auto emitTimings = [&](auto &report_out) {
            report_out << std::fixed << std::endl
                   << " Profiler Output        Time     Percentage" << std::endl
                   << " Timestep              :" << p.compute_dt << " " << 100.0 * (p.compute_dt / elapsed_wall) << std::endl
                   << " Ideal Gas             :" << p.equation_of_state << " " << 100.0 * (p.equation_of_state / elapsed_wall) << std::endl
                   << " Viscosity             :" << p.shock_damping << " " << 100.0 * (p.shock_damping / elapsed_wall) << std::endl
                   << " PdV                   :" << p.PdV << " " << 100.0 * (p.PdV / elapsed_wall) << std::endl
                   << " Revert                :" << p.rollback_state << " " << 100.0 * (p.rollback_state / elapsed_wall) << std::endl
                   << " Acceleration          :" << p.accel_time << " " << 100.0 * (p.accel_time / elapsed_wall) << std::endl
                   << " Fluxes                :" << p.flux_time << " " << 100.0 * (p.flux_time / elapsed_wall) << std::endl
                   << " Cell Advection        :" << p.cell_remap_time << " " << 100.0 * (p.cell_remap_time / elapsed_wall) << std::endl
                   << " Momentum Advection    :" << p.momentum_remap_time << " " << 100.0 * (p.momentum_remap_time / elapsed_wall) << std::endl
                   << " Reset                 :" << p.reset_time << " " << 100.0 * (p.reset_time / elapsed_wall) << std::endl
                   << " Summary               :" << p.summary_time << " " << 100.0 * (p.summary_time / elapsed_wall) << std::endl
                   << " Visit                 :" << p.plot_output << " " << 100.0 * (p.plot_output / elapsed_wall) << std::endl
                   << " Tile Halo Exchange    :" << p.tile_halo_time << " " << 100.0 * (p.tile_halo_time / elapsed_wall) << std::endl
                   << " Self Halo Exchange    :" << p.local_halo_time << " " << 100.0 * (p.local_halo_time / elapsed_wall) << std::endl
                   << " MPI Halo Exchange     :" << p.rank_halo_time << " " << 100.0 * (p.rank_halo_time / elapsed_wall) << std::endl
                   << " Total                 :" << kernels_sum << " " << 100.0 * (kernels_sum / elapsed_wall) << std::endl
                   << " The Rest              :" << elapsed_wall - kernels_sum << " " << 100.0 * (elapsed_wall - kernels_sum) / elapsed_wall
                   << std::endl
                   << std::endl;
          };
          emitTimings(glb_log);
          emitTimings(std::cout);
        }
      }


      break;
    }

    if (!parallel.is_root) continue;

    elapsed_wall = wall_seconds() - run_start;
    double last_step_secs = wall_seconds() - step_start;
    glb_log << "Wall clock " << elapsed_wall << std::endl;
    std::cout << " Wall clock " << elapsed_wall << std::endl;
    double cell_total = globals.settings.mesh_spec.grid_nx * globals.settings.mesh_spec.grid_ny;
    double steps_real = globals.cycle_index;
    double per_cell_time = elapsed_wall / (steps_real * cell_total);
    double per_cell_step_time = last_step_secs / cell_total;
    std::cout << " Average time per cell " << per_cell_time << std::endl;
    glb_log << "Average time per cell " << per_cell_time << std::endl;
    std::cout << "  Step time per cell    " << per_cell_step_time << std::endl;
    glb_log << "Step time per cell    " << per_cell_step_time << std::endl;
  }
}
