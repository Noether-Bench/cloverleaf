
#pragma once

#include <cstdlib>
#ifdef NO_MPI

  #define NET_OK (0)
  #define NET_FAIL_GROUP (1)
  #define NET_FAIL_LEN (2)
  #define NET_FAIL_KIND (3)
  #define NET_FAIL_BUF (4)

  #define NET_INT32 (0)
  #define NET_FLOAT64 (0)
  #define NET_OP_TOTAL (0)
  #define NET_OP_LOWEST (0)
  #define NET_OP_HIGHEST (0)
  #define NET_NO_STATUS (0)

  #define NET_GROUP_ALL (0)

using Net_Group = int;
using Net_Handle = int;
using Net_ElemKind = int;
using Net_Combiner = int;
using Net_Outcome = int;

int MPI_Startup(int *arg_count, char ***arg_values);
int Net_Group_index(Net_Group process_group, int *proc_index);
int Net_Group_count(Net_Group process_group, int *size);
int Net_Terminate(Net_Group process_group, int exit_status);
int Net_SyncPoint(Net_Group process_group);
int MPI_Teardown();

int Net_CombineToRoot(const void *out_buf, void *in_buf, int elem_count, Net_ElemKind elem_kind, Net_Combiner op, int home_proc, Net_Group process_group);
int Net_CombineEvery(const void *out_buf, void *in_buf, int elem_count, Net_ElemKind elem_kind, Net_Combiner op, Net_Group process_group);
int Net_PostSend(const void *buf, int elem_count, Net_ElemKind elem_kind, int target_proc, int tag, Net_Group process_group, Net_Handle *msg_handle);
int Net_PostReceive(void *buf, int elem_count, Net_ElemKind elem_kind, int origin_proc, int tag, Net_Group process_group, Net_Handle *msg_handle);
int Net_GatherEvery(const void *out_buf, int out_count, Net_ElemKind out_kind, void *in_buf, int in_count, Net_ElemKind in_kind,
                  Net_Group process_group);
int Net_AwaitEvery(int elem_count, Net_Handle pending_handles[], Net_Outcome handle_outcomes[]);

#endif