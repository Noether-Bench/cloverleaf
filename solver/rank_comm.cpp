#include "rank_comm.h"

#include <cstdlib>

mpi_context::mpi_context() {

  parallel = true;
  Net_Group_index(NET_GROUP_ALL, &rank_id);
  Net_Group_count(NET_GROUP_ALL, &rank_count);

  is_root = rank_id == 0;
}

void hydro2d_terminate() { Net_Terminate(NET_GROUP_ALL, EXIT_FAILURE); }

void hydro2d_sync() { Net_SyncPoint(NET_GROUP_ALL); }

void hydro2d_sync(sim_state &globals) {
  hydro2d_sync();
}

void hydro2d_reduce_sum(double &local_val) {
  double global_sum = local_val;
  Net_CombineToRoot(&local_val, &global_sum, 1, NET_FLOAT64, NET_OP_TOTAL, 0, NET_GROUP_ALL);
  local_val = global_sum;
}

void hydro2d_reduce_min(double &local_val) {
  double global_min = local_val;
  Net_CombineEvery(&local_val, &global_min, 1, NET_FLOAT64, NET_OP_LOWEST, NET_GROUP_ALL);
  local_val = global_min;
}

void hydro2d_gather_all(double local_val, std::vector<double> &gathered) {
  gathered[0] = local_val;
  Net_GatherEvery(&local_val, 1, NET_FLOAT64, gathered.data(), 1, NET_FLOAT64, NET_GROUP_ALL);
}

int argmax_index(const std::vector<double> &rank_values, const int len) {
  int loc = -1;
  double max = -1.0;
  for (int i = 0; i < len; ++i) {
    if (rank_values[i] >= max) {
      loc = i;
      max = rank_values[i];
    }
  }
  return loc;
}

void hydro2d_max_error(int &error) {
  int max_err = error;
  Net_CombineEvery(&error, &max_err, 1, NET_INT32, NET_OP_HIGHEST, NET_GROUP_ALL);
  error = max_err;
}
