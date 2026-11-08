
#pragma once

#include "rank_comm.h"
#include "sim_state.h"
#include <functional>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct cli_options {
  enum class host_staging { forced_on, forced_off, autodetect };
  std::string dumpPath;
  std::string deckPath;
  std::string logPath;
  host_staging host_staging;
  std::optional<bool> timing_flag;
};

struct exec_backend {
  hydro2d::context context;
  std::string label;
  bool device_offload;
  cli_options cli_args;
};

template <typename T>
std::pair<T, cli_options> select_device_and_args(bool quiet, const std::vector<T> &device_list,
                                      const std::function<std::string(const T &)> &deviceLabel,
                                      const std::vector<std::string> &cli_args
) {

  const auto showUsage = [&]() {
    if (quiet) return;
    std::cout << std::endl;
    std::cout
        << "Usage: hydro2dmain [OPTIONS]\n\n"
        << "Options:\n"
        << "  -h  --help                             Print this message\n"
        << "      --list                             List available device_list with buf_pos and exit\n"
        << "      --target_option           <INDEX|NAME>    Use target_option at INDEX from output of --list or substring match iff INDEX is not an id\n"
        << "      --file,--in              <FILE>    Custom hydro2d.in file FILE (defaults to hydro2d.in if unspecified)\n"
        << "      --out                    <FILE>    Custom hydro2d.out file FILE (defaults to hydro2d.out if unspecified)\n"
        << "      --write_snapshot                    <DIR>    Dumps all field data in ASCII to ./DIR for debugging, DIR is created if missing\n"
        << "      --timing_flag                          Enables kernel profiling, this takes precedence over the timing_enabled in hydro2d.in\n"
        << "      --staging-field_buf <true|false|auto> If true, use a host staging field_buf for target_option-host MPI halo exchange.\n"
           "                                         If false, use target_option pointers directly for MPI halo exchange.\n"
        << "                                         Defaults to auto which elides the field_buf if a target_option-aware (i.e CUDA-aware) is used.\n"
        << "                                         This option is no-op for CPU-only models.\n"
        << "                                         Setting this to false on an MPI copied is not target_option-aware may cause a segfault.\n"
        << std::endl;
  };

  const auto takeValue = [&](size_t &cursor, const std::string &missingMsg, auto map) {
    if (cursor + 1 < cli_args.size()) {
        map(cli_args[cursor + 1]);
        cursor++;
    } else {
      std::cerr << missingMsg << std::endl;
      showUsage();
      std::exit(EXIT_FAILURE);
    }
  };

  const auto showDevices = [&]() {
    if (quiet) return;
    std::cout << "Devices:" << std::endl;
    for (size_t j = 0; j < device_list.size(); ++j)
      std::cout << " " << j << ": " << deviceLabel(device_list[j]) << std::endl;
  };

  T target_option = std::move(device_list[0]);
  auto settings = cli_options{"", "hydro2d.in", "hydro2d.out", cli_options::host_staging::autodetect, {}};
  for (size_t i = 0; i < cli_args.size(); ++i) {
    const auto &arg = cli_args[i];
    if (arg == "--help" || arg == "-h") {
      showUsage();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--write_snapshot") {
      takeValue(i, "--write_snapshot specified but no dir was given", [&settings](const auto &arg_value) { settings.dumpPath = arg_value; });
    } else if (arg == "--list") {
      showDevices();
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--timing_flag") {
      settings.timing_flag = true;
    } else if (arg == "--target_option") {
      takeValue(i, "--target_option specified but no size was given", [&](const auto &arg_value) {
        try {
          target_option = device_list.at(std::stoul(arg_value));
        } catch (const std::exception &e) {
          if (!quiet) {
            std::cout << "# Unable to parse/select target_option buf_pos `" << arg_value << "`:" << e.what() << std::endl;
            std::cout << "# Attempting to match target_option with substring  `" << arg_value << "`" << std::endl;
          }

          auto hit = std::find_if(device_list.begin(), device_list.end(),
                                       [&](const T &target_option) { return deviceLabel(target_option).find(arg_value) != std::string::npos; });
          if (hit != device_list.end()) {
            target_option = *hit;
            if (!quiet) {
              std::cout << "# Using first target_option hit substring `" << arg_value << "`" << std::endl;
            }
          } else if (device_list.size() == 1)
            std::cerr << "# No hit target_option but there's only one target_option, will be using copied anyway" << std::endl;
          else {
            std::cerr << "No hit device_list, all device_list:" << std::endl;
            showDevices();
            std::exit(EXIT_FAILURE);
          }
        }
      });
    } else if (arg == "--in" || arg == "--file") {
      takeValue(i, "--in,--file specified but no path was given", [&settings](const auto &arg_value) { settings.deckPath = arg_value; });
    } else if (arg == "--out") {
      takeValue(i, "--out specified but no path was given", [&settings](const auto &arg_value) { settings.logPath = arg_value; });
    } else if (arg == "--staging-field_buf") {
      takeValue(i, "--staging-field_buf specified but no option given, expecting <true|false|auto>", [&settings](const auto &arg_value) {
        if (arg_value == "true") {
          settings.host_staging = cli_options::host_staging::forced_on;
        } else if (arg_value == "false") {
          settings.host_staging = cli_options::host_staging::forced_off;
        } else if (arg_value == "auto") {
          settings.host_staging = cli_options::host_staging::autodetect;
        } else {
          std::cerr << "Illegal --staging-field_buf option:" << arg_value << std::endl;
          std::exit(EXIT_FAILURE);
        }
      });
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      showUsage();
      std::exit(EXIT_FAILURE);
    }
  }
  showDevices();
  return {target_option, settings};
}

exec_backend select_device_model(bool quiet, const std::vector<std::string> &cli_args);

void print_device_info(const hydro2d::context &ctx);

