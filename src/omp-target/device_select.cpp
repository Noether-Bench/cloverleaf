

#include "device_select.h"
#include "input_deck.h"

#include <algorithm>
#include <omp.h>
#include <sstream>
#include <string>

exec_backend select_device_model(bool quiet, const std::vector<std::string> &cli_args) {

  using target_option = std::pair<std::string, int>;
  auto device_total = omp_get_num_devices();
  std::vector<target_option> device_list(device_total);
  for (size_t i = 0; i < device_list.size(); ++i)
    device_list[i] = {"target target_option (target:true) #" + std::to_string(i), i};

#ifdef OFFLOAD_IF_ENABLED
  device_list.emplace_back("host target_option (target:false)", -1);
#endif

  auto [chosen_device, cli_values] = select_device_and_args<target_option>(
      quiet, device_list, [](const auto &d) { return d.first; }, cli_args);

  if (chosen_device.second != -1) {
    omp_set_default_device(chosen_device.second);
  }

  return exec_backend{hydro2d::context{.use_target = chosen_device.second != -1}, "OpenMP Target", chosen_device.second != -1, cli_values};
}

void print_device_info(const hydro2d::context &ctx) {
  std::cout << " - Device: #" << omp_get_default_device() << ")"
            << " - Target: " << ((!(ctx.use_target) ? ("false") : ("true"))) << std::endl;
}
