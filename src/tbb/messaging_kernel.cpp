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

//  @brief Communication Utilities
//  @author Wayne Gaudin
//  @details Contains all utilities required to run Hydro2dMain in a distributed
//  environment, including initialisation, mesh decompostion, reductions and
//  halo exchange using explicit buffers.
//
//  Note the halo exchange is currently coded as simply as possible and no
//  optimisations have been implemented, such as post receives before sends or packing
//  buffers with multiple data exchange_mask. This is intentional so the effect of these
//  optimisations can be measured on large systems, as and when they are added.
//
//  Even without these modifications Hydro2dMain weak scales well on moderately sized
//  systems of the order of 10K cores.

#include "messaging_kernel.h"
#include "messaging.h"
#include "halo_pack.h"

void hydro2d_alloc_comm_buffers(sim_state &globals, mpi_context &parallel) {

  // Unallocated buffers for external boundaries caused issues on some systems so they are now
  //  all allocated
  if (parallel.rank_id == globals.subdomain.rank_id) {
  }
}

void hydro2d_halo_swap(sim_state &globals, const int exchange_mask[FIELD_COUNT], const int halo_width) {

  // Assuming 1 patch per rank_id, this will be changed

  int horiz_halo_offset[FIELD_COUNT];
  int vert_halo_offset[FIELD_COUNT];

  Net_Handle msg_handle[4] = {0};
  int pending_requests = 0;

  int packed_len_horizontal = 0;
  int packed_len_vertical = 0;
  for (int field = 0; field < FIELD_COUNT; ++field) {
    if (exchange_mask[field] == 1) {
      horiz_halo_offset[field] = packed_len_horizontal;
      vert_halo_offset[field] = packed_len_vertical;
      packed_len_horizontal += halo_width * (globals.subdomain.y_hi + 5);
      packed_len_vertical += halo_width * (globals.subdomain.x_hi + 5);
    }
  }

  static hydro2d::DeviceArray1D<double> left_rcv_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> left_snd_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_rcv_buffer(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_snd_buffer(globals.context, packed_len_horizontal);

  static hydro2d::DeviceArray1D<double> top_rcv_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> top_snd_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_rcv_buffer(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_snd_buffer(globals.context, packed_len_vertical);

  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
    // do first_col exchanges
    // Find first_col hand tile_list
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] == 1) {
        hydro2d_pack_xlo(globals, left_snd_buffer, tile_id, exchange_mask, halo_width, horiz_halo_offset);
      }
    }

    // send and recv messages to the first_col
    hydro2d_post_msgs_left(globals, left_snd_buffer, left_rcv_buffer, packed_len_horizontal, 1, 2, msg_handle[pending_requests],
                                  msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
    // do last_col exchanges
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] == 1) {
        hydro2d_pack_xhi(globals, right_snd_buffer, tile_id, exchange_mask, halo_width, horiz_halo_offset);
      }
    }

    // send message to the last_col
    hydro2d_post_msgs_right(globals, right_snd_buffer, right_rcv_buffer, packed_len_horizontal, 2, 1, msg_handle[pending_requests],
                                   msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  // make a call to wait / sync
  Net_AwaitEvery(pending_requests, msg_handle, NET_NO_STATUS);

  // Copy back to the target_option

  // unpack in first_col sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] == 1) {
        hydro2d_unpack_xlo(globals, left_rcv_buffer, exchange_mask, tile_id, halo_width, horiz_halo_offset);
      }
    }
  }

  // unpack in last_col sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] == 1) {
        hydro2d_unpack_xhi(globals, right_rcv_buffer, exchange_mask, tile_id, halo_width, horiz_halo_offset);
      }
    }
  }

  pending_requests = 0;
  for (Net_Handle &i : msg_handle)
    i = {};

  if (globals.subdomain.subdomain_adjacent[subdomain_ylo] != mesh_edge) {
    // do first_row exchanges
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_ylo] == 1) {
        hydro2d_pack_ylo(globals, bottom_snd_buffer, tile_id, exchange_mask, halo_width, vert_halo_offset);
      }
    }

    // send message downwards
    hydro2d_post_msgs_bottom(globals, bottom_snd_buffer, bottom_rcv_buffer, packed_len_vertical, 3, 4, msg_handle[pending_requests],
                                    msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
    // do top exchanges
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] == 1) {
        hydro2d_pack_yhi(globals, top_snd_buffer, tile_id, exchange_mask, halo_width, vert_halo_offset);
      }
    }

    // send message upwards
    hydro2d_post_msgs_top(globals, top_snd_buffer, top_rcv_buffer, packed_len_vertical, 4, 3, msg_handle[pending_requests],
                                 msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  // need to make a call to wait / sync
  Net_AwaitEvery(pending_requests, msg_handle, NET_NO_STATUS);

  // Copy back to the target_option

  // unpack in top sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] == 1) {
        hydro2d_unpack_yhi(globals, top_rcv_buffer, exchange_mask, tile_id, halo_width, vert_halo_offset);
      }
    }
  }

  // unpack in first_row sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_ylo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_ylo] == 1) {
        hydro2d_unpack_ylo(globals, bottom_rcv_buffer, exchange_mask, tile_id, halo_width, vert_halo_offset);
      }
    }
  }
}

void hydro2d_post_msgs_left(sim_state &globals, hydro2d::DeviceArray1D<double> &left_snd_buffer,
                                   hydro2d::DeviceArray1D<double> &left_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                   Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int left_rank = globals.subdomain.subdomain_adjacent[subdomain_xlo] - 1;
  Net_PostSend(left_snd_buffer.data, message_len, NET_FLOAT64, left_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(left_rcv_buffer.data, message_len, NET_FLOAT64, left_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_right(sim_state &globals, hydro2d::DeviceArray1D<double> &right_snd_buffer,
                                    hydro2d::DeviceArray1D<double> &right_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                    Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int right_rank = globals.subdomain.subdomain_adjacent[subdomain_xhi] - 1;
  Net_PostSend(right_snd_buffer.data, message_len, NET_FLOAT64, right_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(right_rcv_buffer.data, message_len, NET_FLOAT64, right_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_top(sim_state &globals, hydro2d::DeviceArray1D<double> &top_snd_buffer,
                                  hydro2d::DeviceArray1D<double> &top_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                  Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int top_rank = globals.subdomain.subdomain_adjacent[subdomain_yhi] - 1;
  Net_PostSend(top_snd_buffer.data, message_len, NET_FLOAT64, top_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(top_rcv_buffer.data, message_len, NET_FLOAT64, top_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_bottom(sim_state &globals, hydro2d::DeviceArray1D<double> &bottom_snd_buffer,
                                     hydro2d::DeviceArray1D<double> &bottom_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                     Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  int bottom_rank = globals.subdomain.subdomain_adjacent[subdomain_ylo] - 1;
  Net_PostSend(bottom_snd_buffer.data, message_len, NET_FLOAT64, bottom_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(bottom_rcv_buffer.data, message_len, NET_FLOAT64, bottom_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
