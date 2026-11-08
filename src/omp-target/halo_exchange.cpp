

#include "halo_exchange.h"
#include "rank_comm.h"

#include <algorithm>
#include <array>
#include <iterator>

void hydro2d_post_msgs_left(sim_state &globals, double *left_snd_buffer, double *left_rcv_buffer, int message_len,
                                   int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int left_rank = globals.subdomain.subdomain_adjacent[subdomain_xlo] - 1;
  Net_PostSend(left_snd_buffer, message_len, NET_FLOAT64, left_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(left_rcv_buffer, message_len, NET_FLOAT64, left_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}

void hydro2d_post_msgs_right(sim_state &globals, double *right_snd_buffer, double *right_rcv_buffer, int message_len,
                                    int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int right_rank = globals.subdomain.subdomain_adjacent[subdomain_xhi] - 1;
  Net_PostSend(right_snd_buffer, message_len, NET_FLOAT64, right_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(right_rcv_buffer, message_len, NET_FLOAT64, right_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}

void hydro2d_post_msgs_bottom(sim_state &globals, double *bottom_snd_buffer, double *bottom_rcv_buffer, int message_len,
                                     int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int bottom_rank = globals.subdomain.subdomain_adjacent[subdomain_ylo] - 1;
  Net_PostSend(bottom_snd_buffer, message_len, NET_FLOAT64, bottom_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(bottom_rcv_buffer, message_len, NET_FLOAT64, bottom_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}

void hydro2d_post_msgs_top(sim_state &globals, double *top_snd_buffer, double *top_rcv_buffer, int message_len, int outbound_label,
                                  int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int top_rank = globals.subdomain.subdomain_adjacent[subdomain_yhi] - 1;
  Net_PostSend(top_snd_buffer, message_len, NET_FLOAT64, top_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(top_rcv_buffer, message_len, NET_FLOAT64, top_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}

void hydro2d_halo_swap(sim_state &globals, const int exchange_mask[FIELD_COUNT], const int halo_width) {


  int horiz_halo_offset[FIELD_COUNT];
  int vert_halo_offset[FIELD_COUNT];

  std::array<Net_Handle, 4> msg_handle{};
  int pending_requests = 0;

  int packed_len_horizontal = 0;
  int packed_len_vertical = 0;
  for (int field = 0; field < FIELD_COUNT; ++field) {
    if (exchange_mask[field] != 1) continue;
    horiz_halo_offset[field] = packed_len_horizontal;
    vert_halo_offset[field] = packed_len_vertical;
    packed_len_horizontal += halo_width * (globals.subdomain.y_hi + 5);
    packed_len_vertical += halo_width * (globals.subdomain.x_hi + 5);
  }

  static hydro2d::DeviceArray1D<double> left_rcv_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> left_snd_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_rcv_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_snd_buffer(globals.context, packed_len_horizontal);

  static hydro2d::DeviceArray1D<double> top_rcv_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> top_snd_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_rcv_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_snd_buffer(globals.context, packed_len_vertical);

  double *left_rcv = left_rcv_buffer.data;
  double *left_snd = left_snd_buffer.data;
  double *right_rcv = right_rcv_buffer.data;
  double *right_snd = right_snd_buffer.data;
  double *top_rcv = top_rcv_buffer.data;
  double *top_snd = top_snd_buffer.data;
  double *bottom_rcv = bottom_rcv_buffer.data;
  double *bottom_snd = bottom_snd_buffer.data;

  bool stage = globals.settings.host_staging;

#pragma omp target enter data map(alloc : left_rcv[ : left_rcv_buffer.N()]) map(alloc : left_snd[ : left_snd_buffer.N()])                  \
    map(alloc : right_rcv[ : right_rcv_buffer.N()]) map(alloc : right_snd[ : right_snd_buffer.N()])                                        \
    map(alloc : top_rcv[ : top_rcv_buffer.N()]) map(alloc : top_snd[ : top_snd_buffer.N()])                                                \
    map(alloc : bottom_rcv[ : bottom_rcv_buffer.N()]) map(alloc : bottom_snd[ : bottom_snd_buffer.N()])

  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] != 1) continue;
      hydro2d_pack_xlo(globals, left_snd_buffer, tile_id, exchange_mask, halo_width, horiz_halo_offset);
    }

#pragma omp target update if (stage) from(left_snd[ : left_snd_buffer.N()])
#pragma omp target data if (!stage) use_device_ptr(left_snd, left_rcv)
    hydro2d_post_msgs_left(globals, left_snd, left_rcv, packed_len_horizontal, 1, 2, msg_handle[pending_requests],
                                  msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] != 1) continue;
      hydro2d_pack_xhi(globals, right_snd_buffer, tile_id, exchange_mask, halo_width, horiz_halo_offset);
    }

#pragma omp target update if (stage) from(right_snd[ : right_snd_buffer.N()])
#pragma omp target data if (!stage) use_device_ptr(right_snd, right_rcv)
    hydro2d_post_msgs_right(globals, right_snd, right_rcv, packed_len_horizontal, 2, 1, msg_handle[pending_requests],
                                   msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  Net_AwaitEvery(pending_requests, msg_handle.data(), NET_NO_STATUS);


  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
#pragma omp target update if (stage) to(left_rcv[ : left_rcv_buffer.N()])
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] != 1) continue;
      hydro2d_unpack_xlo(globals, left_rcv_buffer, exchange_mask, tile_id, halo_width, horiz_halo_offset);
    }
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
#pragma omp target update if (stage) to(right_rcv[ : right_rcv_buffer.N()])
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] != 1) continue;
      hydro2d_unpack_xhi(globals, right_rcv_buffer, exchange_mask, tile_id, halo_width, horiz_halo_offset);
    }
  }

  pending_requests = 0;
  std::for_each(std::begin(msg_handle), std::end(msg_handle), [](Net_Handle &handle) { handle = {}; });

  if (globals.subdomain.subdomain_adjacent[subdomain_ylo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_ylo] != 1) continue;
      hydro2d_pack_ylo(globals, bottom_snd_buffer, tile_id, exchange_mask, halo_width, vert_halo_offset);
    }

#pragma omp target update if (stage) from(bottom_snd[ : bottom_snd_buffer.N()])
#pragma omp target data if (!stage) use_device_ptr(bottom_snd, bottom_rcv)
    hydro2d_post_msgs_bottom(globals, bottom_snd, bottom_rcv, packed_len_vertical, 3, 4, msg_handle[pending_requests],
                                    msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] != 1) continue;
      hydro2d_pack_yhi(globals, top_snd_buffer, tile_id, exchange_mask, halo_width, vert_halo_offset);
    }

#pragma omp target update if (stage) from(top_snd[ : top_snd_buffer.N()])
#pragma omp target data if (!stage) use_device_ptr(top_snd, top_rcv)
    hydro2d_post_msgs_top(globals, top_snd, top_rcv, packed_len_vertical, 4, 3, msg_handle[pending_requests],
                                 msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  Net_AwaitEvery(pending_requests, msg_handle.data(), NET_NO_STATUS);


  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
#pragma omp target update to(top_rcv[ : top_rcv_buffer.N()]) if (stage)
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] != 1) continue;
      hydro2d_unpack_yhi(globals, top_rcv_buffer, exchange_mask, tile_id, halo_width, vert_halo_offset);
    }
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_ylo] != mesh_edge) {
#pragma omp target update to(bottom_rcv[ : bottom_rcv_buffer.N()]) if (stage)
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_ylo] != 1) continue;
      hydro2d_unpack_ylo(globals, bottom_rcv_buffer, exchange_mask, tile_id, halo_width, vert_halo_offset);
    }
  }

#pragma omp target exit data map(release : left_rcv[ : 0]) map(release : left_snd[ : 0]) map(release : right_rcv[ : 0])                    \
    map(release : right_snd[ : 0]) map(release : top_rcv[ : 0]) map(release : top_snd[ : 0]) map(release : bottom_rcv[ : 0])               \
    map(release : bottom_snd[ : 0])
}
