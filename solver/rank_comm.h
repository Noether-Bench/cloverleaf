
#pragma once

#include "device_context.h"
#include "sim_state.h"

#ifdef NO_MPI
  #include "mpi_stub.h"
#else
  #define OMPI_NO_CXX_BINDINGS
  #include <mpi.h>

  #if __has_include("mpi-ext.h")
    #include "mpi-ext.h"
  #endif

#endif

struct mpi_context {

  bool parallel;

  bool is_root;

  int rank_count{};

  int rank_id{};

  mpi_context();
};

void hydro2d_terminate();
void hydro2d_sync(sim_state &globals);
void hydro2d_sync();

void hydro2d_reduce_sum(double &local_val);
void hydro2d_reduce_min(double &local_val);
void hydro2d_gather_all(double local_val, std::vector<double> &gathered);
int argmax_index(const std::vector<double> &rank_values, int len);
void hydro2d_max_error(int &error);
