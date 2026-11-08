
#include "mpi_stub.h"
#include <cstdio>

#ifdef NO_MPI

int MPI_Startup(int *, char ***) { return NET_OK; }
int Net_Group_index(Net_Group, int *proc_index) {
  *proc_index = 0;
  return NET_OK;
}
int Net_Group_count(Net_Group, int *size) {
  *size = 1;
  return NET_OK;
}
int Net_Terminate(Net_Group, int exit_status) {
  std::exit(exit_status);
  return NET_OK;
}
int MPI_Teardown() { return NET_OK; }

int Net_SyncPoint(Net_Group) {
  return NET_OK;
}
int Net_GatherEvery(const void *, int, Net_ElemKind, void *, int, Net_ElemKind, Net_Group) {
  return NET_OK;
}
int Net_CombineToRoot(const void *, void *, int, Net_ElemKind, Net_Combiner, int, Net_Group) {
  return NET_OK;
}
int Net_CombineEvery(const void *, void *, int, Net_ElemKind, Net_Combiner, Net_Group) {
  return NET_OK;
}
int Net_AwaitEvery(int, Net_Handle[], Net_Outcome[]) {
  return NET_OK;
}

int Net_PostSend(const void *, int, Net_ElemKind, int, int, Net_Group, Net_Handle *) {
  fprintf(stderr, "MPI disabled, stub: %s\n", __func__);
  std::abort();
  return NET_FAIL_GROUP;
}
int Net_PostReceive(void *, int, Net_ElemKind, int, int, Net_Group, Net_Handle *) {
  fprintf(stderr, "MPI disabled, stub: %s\n", __func__);
  std::abort();
  return NET_FAIL_GROUP;
}

#endif
