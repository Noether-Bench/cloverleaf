

#include <iostream>

#include "rank_comm.h"
#include "sim_state.h"
#include "field_storage.h"
#include "run_loop.h"
#include "device_select.h"
#include "input_deck.h"
#include "run_log.h"
#include "problem_setup.h"

#define glb_release_number "2.000"

std::ostream glb_log(nullptr);

std::ofstream of;

sim_state startup(mpi_context &parallel, const std::vector<std::string> &cli_args) {
  run_config settings;
  if (parallel.is_root) {
    std::cout << "---" << std::endl;
  }
  auto exec_backend = select_device_model(!parallel.is_root, cli_args);
  settings.dumpPath = exec_backend.cli_args.dumpPath;

#ifdef NO_MPI
  bool mpi_active = false;
#else
  bool mpi_active = true;
#endif

#if defined(MPIX_GPU_AWARE_BUILD) && MPIX_GPU_AWARE_BUILD
  std::optional<bool> mpi_gpu_aware_compiled = true;
#elif defined(MPIX_GPU_AWARE_BUILD) && !MPIX_GPU_AWARE_BUILD
  std::optional<bool> mpi_gpu_aware_compiled = false;
#else
  std::optional<bool> mpi_gpu_aware_compiled = {};
#endif

#if defined(MPIX_GPU_AWARE_BUILD)
  std::optional<bool> mpi_gpu_aware_detected = MPIX_Query_gpu_aware() != 0;
#else
  std::optional<bool> mpi_gpu_aware_detected = {};
#endif

  if (!exec_backend.device_offload && exec_backend.cli_args.host_staging == cli_options::host_staging::forced_on) {
    std::cout << "# WARNING: enabling staging field_buf on a non-device_offload (host) exec_backend or target_option may be no-op" << std::endl;
  }
  switch (exec_backend.cli_args.host_staging) {
    case cli_options::host_staging::forced_on: settings.host_staging = true; break;
    case cli_options::host_staging::forced_off: settings.host_staging = false; break;
    case cli_options::host_staging::autodetect:
      settings.host_staging = !(mpi_gpu_aware_compiled.value_or(false) && mpi_gpu_aware_detected.value_or(false));
      break;
  }

  if (parallel.is_root) {
    std::cout << "Hydro2dMain:\n"
              << " - Ver.:     " << glb_release_number << "\n"
              << " - Deck:     " << exec_backend.cli_args.deckPath << "\n"
              << " - Out:      " << exec_backend.cli_args.logPath << "\n"
              << " - Profiler: " << (exec_backend.cli_args.timing_flag ? (*exec_backend.cli_args.timing_flag ? "true" : "false") : "deck-specified") << "\n"
              << "MPI:\n"
              << " - Enabled:     " << (mpi_active ? "true" : "false") << "\n"
              << " - Total ranks: " << parallel.rank_count << "\n"
              << " - Header target_option-awareness (CUDA-awareness):  "
              << (mpi_gpu_aware_compiled ? (*mpi_gpu_aware_compiled ? "true" : "false") : "unknown") << "\n"
              << " - Runtime target_option-awareness (CUDA-awareness): "
              << (mpi_gpu_aware_detected ? (*mpi_gpu_aware_detected ? "true" : "false") : "unknown") << "\n"
              << " - Host-Device halo exchange staging field_buf:  " << (settings.host_staging ? "true" : "false") << "\n"
              << "Model:\n"
              << " - Name:      " << exec_backend.label << "\n"
              << " - Execution: " << (exec_backend.device_offload ? "Offload (target_option)" : "Host")
              << std::endl;
    print_device_info(exec_backend.context);
    std::cout << "# ---- " << std::endl;
    std::cout << "Output: |+1" << std::endl;
  }

  if (parallel.is_root) {
    std::cout << " Output file hydro2d.out opened. All output will go there." << std::endl;
    of.open(exec_backend.cli_args.logPath.empty() ? "hydro2d.out" : exec_backend.cli_args.logPath);
    if (!of.is_open()) fatal_message((char *)"startup", (char *)"Error opening hydro2d.out file.");
    glb_log.rdbuf(of.rdbuf());
  } else {
    glb_log.rdbuf(std::cout.rdbuf());
  }

  if (parallel.is_root) {
    glb_log << "Hydro2d Version " << glb_release_number << std::endl
          << "Task Count " << parallel.rank_count << std::endl
          << std::endl;
  }

  hydro2d_sync();

  std::ifstream glb_input;
  if (parallel.is_root) {
    glb_log << "Hydro2d will run from the following input:-" << std::endl << std::endl;
    if (!cli_args.empty()) {
      std::cout << " Args:";
      for (const auto &arg : cli_args)
        std::cout << " " << arg;
      std::cout << std::endl;
    }
  }

  if (!exec_backend.cli_args.deckPath.empty()) {
    if (parallel.is_root) std::cout << " Using input: `" << exec_backend.cli_args.deckPath << "`" << std::endl;
    glb_input.open(exec_backend.cli_args.deckPath);
    if (glb_input.fail()) {
      std::cerr << "Unable to open file: `" << exec_backend.cli_args.deckPath << "`" << std::endl;
      std::exit(1);
    }
  } else {
    if (parallel.is_root) std::cout << "No input file specified, using default input" << std::endl;
    std::ofstream deck_writer("hydro2d.in");
    deck_writer << "*hydro2d" << std::endl
             << " region_id 1 initial_density=0.2 initial_energy=1.0" << std::endl
             << " region_id 2 initial_density=1.0 initial_energy=2.5 shape_selector=rectangle x_lower=0.0 x_upper=5.0 y_lower=0.0 y_upper=2.0" << std::endl
             << " grid_nx=10" << std::endl
             << " grid_ny=2" << std::endl
             << " x_lower=0.0" << std::endl
             << " y_lower=0.0" << std::endl
             << " x_upper=10.0" << std::endl
             << " y_upper=2.0" << std::endl
             << " initial_timestep=0.04" << std::endl
             << " timestep_rise=1.5" << std::endl
             << " max_timestep=0.04" << std::endl
             << " stop_time=3.0" << std::endl
             << " test_case_id 1" << std::endl
             << "*endhydro2d" << std::endl;
    deck_writer.close();
    glb_input.open("hydro2d.in");
  }

  hydro2d_sync();
  if (parallel.is_root) {
    glb_log << std::endl << "Initialising and generating" << std::endl << std::endl;
  }
  parse_deck(glb_input, parallel, settings);
  if (exec_backend.cli_args.timing_flag) {
    settings.timing_enabled = *exec_backend.cli_args.timing_flag;
  }

  hydro2d_sync();

  settings.total_subdomains = parallel.rank_count;

  auto globals = setup_run(parallel, settings, exec_backend.context);
  hydro2d_sync(globals);
  if (parallel.is_root) {
    glb_log << "Starting the calculation" << std::endl;
  }
  glb_input.close();
  return globals;
}

int main(int arg_count, char *arg_values[]) {

  MPI_Startup(&arg_count, &arg_values);
  mpi_context parallel;
  sim_state settings = startup(parallel, std::vector<std::string>(arg_values + 1, arg_values + arg_count));
  if (parallel.is_root) {
    std::cout << " Launching time_march" << std::endl;
  }
  time_march(settings, parallel);
  shutdown_run(settings);
  MPI_Teardown();

  if (parallel.is_root) {
    std::cout << "Result:\n"
              << " - Problem: " << (settings.settings.test_case_id == 0 ? "none" : std::to_string(settings.settings.test_case_id)) << "\n"
              << " - Outcome: " << (settings.test_failed ? "FAILED" : "PASSED") << std::endl;
  }
  return settings.test_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
