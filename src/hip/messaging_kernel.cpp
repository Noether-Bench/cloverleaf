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

    //		new(&globals.subdomain.left_snd)   Kokkos::View<double *>("left_snd", 10 * 2 * (globals.subdomain.y_hi +	5));
    //		new(&globals.subdomain.left_rcv)   Kokkos::View<double *>("left_rcv", 10 * 2 * (globals.subdomain.y_hi +	5));
    //		new(&globals.subdomain.right_snd)  Kokkos::View<double *>("right_snd", 10 * 2 * (globals.subdomain.y_hi +	5));
    //		new(&globals.subdomain.right_rcv)  Kokkos::View<double *>("right_rcv", 10 * 2 * (globals.subdomain.y_hi +	5));
    //		new(&globals.subdomain.bottom_snd) Kokkos::View<double *>("bottom_snd", 10 * 2 * (globals.subdomain.x_hi +	5));
    //		new(&globals.subdomain.bottom_rcv) Kokkos::View<double *>("bottom_rcv", 10 * 2 * (globals.subdomain.x_hi +	5));
    //		new(&globals.subdomain.top_snd)    Kokkos::View<double *>("top_snd", 10 * 2 * (globals.subdomain.x_hi +	5));
    //		new(&globals.subdomain.top_rcv)    Kokkos::View<double *>("top_rcv", 10 * 2 * (globals.subdomain.x_hi +	5));
    //
    //		// Create host mirrors of target_option buffers. This makes this, and deep_copy, a no-op if the View is in host memory already.
    //		globals.subdomain.hm_left_snd = Kokkos::create_mirror_view(
    //				globals.subdomain.left_snd);
    //		globals.subdomain.hm_left_rcv = Kokkos::create_mirror_view(
    //				globals.subdomain.left_rcv);
    //		globals.subdomain.hm_right_snd = Kokkos::create_mirror_view(
    //				globals.subdomain.right_snd);
    //		globals.subdomain.hm_right_rcv = Kokkos::create_mirror_view(
    //				globals.subdomain.right_rcv);
    //		globals.subdomain.hm_bottom_snd = Kokkos::create_mirror_view(
    //				globals.subdomain.bottom_snd);
    //		globals.subdomain.hm_bottom_rcv = Kokkos::create_mirror_view(
    //				globals.subdomain.bottom_rcv);
    //		globals.subdomain.hm_top_snd = Kokkos::create_mirror_view(globals.subdomain.top_snd);
    //		globals.subdomain.hm_top_rcv = Kokkos::create_mirror_view(globals.subdomain.top_rcv);
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

  static hydro2d::DeviceArray1D<double> left_rcv_device(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> left_snd_device(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_rcv_device(globals.context, packed_len_horizontal);
  static hydro2d::DeviceArray1D<double> right_snd_device(globals.context, packed_len_horizontal);

  static hydro2d::DeviceArray1D<double> top_rcv_device(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> top_snd_device(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_rcv_device(globals.context, packed_len_vertical);
  static hydro2d::DeviceArray1D<double> bottom_snd_device(globals.context, packed_len_vertical);

  bool stage = globals.settings.host_staging;

  static auto left_rcv_staging_vector = std::vector<double>(stage ? left_rcv_device.size : 0);
  static auto left_snd_staging_vector = std::vector<double>(stage ? left_snd_device.size : 0);
  static auto right_rcv_staging_vector = std::vector<double>(stage ? right_rcv_device.size : 0);
  static auto right_snd_staging_vector = std::vector<double>(stage ? right_snd_device.size : 0);

  static auto top_rcv_staging_vector = std::vector<double>(stage ? top_rcv_device.size : 0);
  static auto top_snd_staging_vector = std::vector<double>(stage ? top_snd_device.size : 0);
  static auto bottom_rcv_staging_vector = std::vector<double>(stage ? bottom_rcv_device.size : 0);
  static auto bottom_snd_staging_vector = std::vector<double>(stage ? bottom_snd_device.size : 0);

  auto left_rcv_staging = stage ? left_rcv_staging_vector.data() : nullptr;
  auto left_snd_staging = stage ? left_snd_staging_vector.data() : nullptr;
  auto right_rcv_staging = stage ? right_rcv_staging_vector.data() : nullptr;
  auto right_snd_staging = stage ? right_snd_staging_vector.data() : nullptr;
  auto top_rcv_staging = stage ? top_rcv_staging_vector.data() : nullptr;
  auto top_snd_staging = stage ? top_snd_staging_vector.data() : nullptr;
  auto bottom_rcv_staging = stage ? bottom_rcv_staging_vector.data() : nullptr;
  auto bottom_snd_staging = stage ? bottom_snd_staging_vector.data() : nullptr;

  auto deviceToStaging = [](double *staging, hydro2d::DeviceArray1D<double> &target_option) {
    hydro2d::checkError(hipMemcpy(staging, target_option.data, target_option.size * sizeof(double), CLOVER_MEMCPY_KIND_D2H));
  };

  auto stagingToDevice = [](double *staging, hydro2d::DeviceArray1D<double> &target_option) {
    hydro2d::checkError(hipMemcpy(target_option.data, staging, target_option.size * sizeof(double), CLOVER_MEMCPY_KIND_H2D));
  };

  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
    // do first_col exchanges
    // Find first_col hand tile_list
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] == 1) {
        hydro2d_pack_xlo(globals, left_snd_device, tile_id, exchange_mask, halo_width, horiz_halo_offset);
      }
    }
    if (stage) deviceToStaging(left_snd_staging, left_snd_device);
    else {
      left_snd_staging = left_snd_device.data;
      left_rcv_staging = left_rcv_device.data;
    }
    // send and recv messages to the first_col
    hydro2d_post_msgs_left(globals, left_snd_staging, left_rcv_staging, packed_len_horizontal, 1, 2, msg_handle[pending_requests],
                                  msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
    // do last_col exchanges
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] == 1) {
        hydro2d_pack_xhi(globals, right_snd_device, tile_id, exchange_mask, halo_width, horiz_halo_offset);
      }
    }

    if (stage) deviceToStaging(right_snd_staging, right_snd_device);
    else {
      right_snd_staging = right_snd_device.data;
      right_rcv_staging = right_rcv_device.data;
    }
    // send message to the last_col
    hydro2d_post_msgs_right(globals, right_snd_staging, right_rcv_staging, packed_len_horizontal, 2, 1, msg_handle[pending_requests],
                                   msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  // make a call to wait / sync
  Net_AwaitEvery(pending_requests, msg_handle, NET_NO_STATUS);
  if (stage) {
    stagingToDevice(left_rcv_staging, left_rcv_device);
    stagingToDevice(right_rcv_staging, right_rcv_device);
  }

  // Copy back to the target_option

  // unpack in first_col sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_xlo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[west_tile] == 1) {
        hydro2d_unpack_xlo(globals, left_rcv_device, exchange_mask, tile_id, halo_width, horiz_halo_offset);
      }
    }
  }

  // unpack in last_col sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_xhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_xhi] == 1) {
        hydro2d_unpack_xhi(globals, right_rcv_device, exchange_mask, tile_id, halo_width, horiz_halo_offset);
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
        hydro2d_pack_ylo(globals, bottom_snd_device, tile_id, exchange_mask, halo_width, vert_halo_offset);
      }
    }

    if (stage) deviceToStaging(bottom_snd_staging, bottom_snd_device);
    else {
      bottom_snd_staging = bottom_snd_device.data;
      bottom_rcv_staging = bottom_rcv_device.data;
    }
    // send message downwards
    hydro2d_post_msgs_bottom(globals, bottom_snd_staging, bottom_rcv_staging, packed_len_vertical, 3, 4,
                                    msg_handle[pending_requests], msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
    // do top exchanges
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] == 1) {
        hydro2d_pack_yhi(globals, top_snd_device, tile_id, exchange_mask, halo_width, vert_halo_offset);
      }
    }

    if (stage) deviceToStaging(top_snd_staging, top_snd_device);
    else {
      top_snd_staging = top_snd_device.data;
      top_rcv_staging = top_rcv_device.data;
    }
    // send message upwards
    hydro2d_post_msgs_top(globals, top_snd_staging, top_rcv_staging, packed_len_vertical, 4, 3, msg_handle[pending_requests],
                                 msg_handle[pending_requests + 1]);
    pending_requests += 2;
  }

  // need to make a call to wait / sync
  Net_AwaitEvery(pending_requests, msg_handle, NET_NO_STATUS);
  if (stage) {
    stagingToDevice(bottom_rcv_staging, bottom_rcv_device);
    stagingToDevice(top_rcv_staging, top_rcv_device);
  }

  // Copy back to the target_option

  // unpack in top sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_yhi] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_yhi] == 1) {
        hydro2d_unpack_yhi(globals, top_rcv_device, exchange_mask, tile_id, halo_width, vert_halo_offset);
      }
    }
  }

  // unpack in first_row sweep_dir
  if (globals.subdomain.subdomain_adjacent[subdomain_ylo] != mesh_edge) {
    for (int tile_id = 0; tile_id < globals.settings.tiles_per_subdomain; ++tile_id) {
      if (globals.subdomain.tile_list[tile_id].meta.edge_mask[tile_ylo] == 1) {
        hydro2d_unpack_ylo(globals, bottom_rcv_device, exchange_mask, tile_id, halo_width, vert_halo_offset);
      }
    }
  }
}

void hydro2d_post_msgs_left(sim_state &globals, double *left_snd_buffer, double *left_rcv_buffer, int message_len,
                                   int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  hydro2d::checkError(hipDeviceSynchronize());
  int left_rank = globals.subdomain.subdomain_adjacent[subdomain_xlo] - 1;
  Net_PostSend(left_snd_buffer, message_len, NET_FLOAT64, left_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(left_rcv_buffer, message_len, NET_FLOAT64, left_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_right(sim_state &globals, double *right_snd_buffer, double *right_rcv_buffer, int message_len,
                                    int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  hydro2d::checkError(hipDeviceSynchronize());
  int right_rank = globals.subdomain.subdomain_adjacent[subdomain_xhi] - 1;
  Net_PostSend(right_snd_buffer, message_len, NET_FLOAT64, right_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(right_rcv_buffer, message_len, NET_FLOAT64, right_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_top(sim_state &globals, double *top_snd_buffer, double *top_rcv_buffer, int message_len, int outbound_label,
                                  int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  hydro2d::checkError(hipDeviceSynchronize());
  int top_rank = globals.subdomain.subdomain_adjacent[subdomain_yhi] - 1;
  Net_PostSend(top_snd_buffer, message_len, NET_FLOAT64, top_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(top_rcv_buffer, message_len, NET_FLOAT64, top_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
void hydro2d_post_msgs_bottom(sim_state &globals, double *bottom_snd_buffer, double *bottom_rcv_buffer, int message_len,
                                     int outbound_label, int inbound_label, Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  hydro2d::checkError(hipDeviceSynchronize());
  int bottom_rank = globals.subdomain.subdomain_adjacent[subdomain_ylo] - 1;
  Net_PostSend(bottom_snd_buffer, message_len, NET_FLOAT64, bottom_rank, outbound_label, NET_GROUP_ALL, &outbound_handle);
  Net_PostReceive(bottom_rcv_buffer, message_len, NET_FLOAT64, bottom_rank, inbound_label, NET_GROUP_ALL, &inbound_handle);
}
