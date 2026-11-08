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

#include <fstream>

#include "startup.h"
#include "setup_run.h"

exec_backend select_device_model(bool quiet, const std::vector<std::string> &cli_args) {
  struct Device {
    int id{};
    std::string label{};
  };
  int elem_count = 0;
  hydro2d::checkError(hipGetDeviceCount(&elem_count));
  std::vector<Device> device_list(elem_count);
  for (int i = 0; i < elem_count; ++i) {
    hipDeviceProp_t props{};
    hydro2d::checkError(hipGetDeviceProperties(&props, i));
    device_list[i] = {i, std::string(props.label) + " (" +                                        //
                         std::to_string(props.totalGlobalMem / 1024 / 1024) + "MB;" +        //
                         "sm_" + std::to_string(props.major) + std::to_string(props.minor) + //
                         ")"};
  }
  auto [target_option, cli_values] = select_device_and_args<Device>(
      quiet, device_list, [](auto &d) { return d.label; }, cli_args);
  hydro2d::checkError(hipSetDevice(target_option.id));
  return exec_backend{hydro2d::context{}, "HIP", true, cli_values};
}

void print_device_info(const hydro2d::context &) {
  int target_option = -1;
  hydro2d::checkError(hipGetDevice(&target_option));
  hipDeviceProp_t props{};
  hydro2d::checkError(hipGetDeviceProperties(&props, target_option));
  std::cout << " - Device: " //
            << props.label << " (" << (props.totalGlobalMem / 1024 / 1024) << "MB;"
            << "gfx" << props.gcnArch << ")" << std::endl;
  std::cout << " - HIP managed memory: "
            <<
#ifdef CLOVER_MANAGED_ALLOC
      "true"
#else
      "false"
#endif
            << std::endl;
  std::cout << " - HIP per-kernel synchronisation: "
            <<
#ifdef CLOVER_SYNC_ALL_KERNELS
      "true"
#else
      "false"
#endif
            << std::endl;
}
