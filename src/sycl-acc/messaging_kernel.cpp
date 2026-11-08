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

#include <sstream>

#include "messaging_kernel.h"
#include "messaging.h"
#include "context.h"
#include "halo_pack.h"


void hydro2d_alloc_comm_buffers(sim_state &globals, mpi_context &parallel) {

  // Unallocated buffers for external boundaries caused issues on some systems so they are now
  //  all allocated
  if (parallel.rank_id == globals.subdomain.rank_id) {

    //		new(&globals.subdomain.left_snd_buffer)   Kokkos::View<double *>("left_snd_buffer", 10 * 2 *
    //(globals.subdomain.y_hi +	5)); 		new(&globals.subdomain.left_rcv_buffer)   Kokkos::View<double
    //*>("left_rcv_buffer", 10 * 2 * (globals.subdomain.y_hi +	5)); 		new(&globals.subdomain.right_snd_buffer)
    // Kokkos::View<double *>("right_snd_buffer", 10
    //* 2 * (globals.subdomain.y_hi +	5)); 		new(&globals.subdomain.right_rcv_buffer)  Kokkos::View<double
    //*>("right_rcv_buffer", 10 * 2 * (globals.subdomain.y_hi +	5)); 		new(&globals.subdomain.bottom_snd_buffer)
    // Kokkos::View<double *>("bottom_snd_buffer", 10 * 2 * (globals.subdomain.x_hi +	5));
    //		new(&globals.subdomain.bottom_rcv_buffer) Kokkos::View<double *>("bottom_rcv_buffer", 10 * 2 *
    //(globals.subdomain.x_hi +	5)); 		new(&globals.subdomain.top_snd_buffer)    Kokkos::View<double
    //*>("top_snd_buffer", 10
    //* 2 * (globals.subdomain.x_hi +	5)); 		new(&globals.subdomain.top_rcv_buffer)    Kokkos::View<double
    //*>("top_rcv_buffer", 10 * 2 * (globals.subdomain.x_hi +	5));
    //
    //		// Create host mirrors of target_option buffers. This makes this, and deep_copy, a no-op if the View is in host
    // memory already. 		globals.subdomain.hm_left_snd_buffer = Kokkos::create_mirror_view(
    // globals.subdomain.left_snd_buffer); 		globals.subdomain.hm_left_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.subdomain.left_rcv_buffer);
    //		globals.subdomain.hm_right_snd_buffer = Kokkos::create_mirror_view(
    //				globals.subdomain.right_snd_buffer);
    //		globals.subdomain.hm_right_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.subdomain.right_rcv_buffer);
    //		globals.subdomain.hm_bottom_snd_buffer = Kokkos::create_mirror_view(
    //				globals.subdomain.bottom_snd_buffer);
    //		globals.subdomain.hm_bottom_rcv_buffer = Kokkos::create_mirror_view(
    //				globals.subdomain.bottom_rcv_buffer);
    //		globals.subdomain.hm_top_snd_buffer = Kokkos::create_mirror_view(globals.subdomain.top_snd_buffer);
    //		globals.subdomain.hm_top_rcv_buffer = Kokkos::create_mirror_view(globals.subdomain.top_rcv_buffer);
  }
}
#if !(defined(__HIPSYCL__) || defined(__OPENSYCL__))
template <typename A> decltype(auto) get_native_ptr_or_throw(sycl::interop_handle &ih, A accessor) {
  using sycl::backend;
  using T = std::remove_cv_t<typename decltype(accessor)::value_type>;
  switch (ih.get_backend()) {
    case backend::ext_oneapi_level_zero: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_level_zero>(accessor));
  #ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    case backend::ext_oneapi_cuda: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_cuda>(accessor));
  #endif
  #ifdef SYCL_EXT_ONEAPI_BACKEND_HIP
    case backend::ext_oneapi_hip: return reinterpret_cast<T *>(ih.get_native_mem<backend::ext_oneapi_hip>(accessor));
  #endif
    default:
      std::stringstream ss;
      ss << "backend " << ih.get_backend() << " does not support a pointer-based sycl::interop_handle::get_native_mem";
      throw std::logic_error(ss.str());
  }
}
#endif

void clover_send_recv_message(sim_state &globals, subdomain_side_kind tpe, hydro2d::DeviceArray1D<double> &snd_buffer,
                              hydro2d::DeviceArray1D<double> &rcv_buffer, int message_len, int outbound_label, int inbound_label, Net_Handle &outbound_handle,
                              Net_Handle &inbound_handle) {
  int rank_id = globals.subdomain.subdomain_adjacent[tpe] - 1;
#ifdef USE_HOSTTASK
  if (globals.settings.host_staging) {
    globals.context.exec_ctx.submit([&](sycl::handler &h) {
      auto snd_buffer_acc = snd_buffer.field_buf.get_host_access(h, sycl::read_only);
      auto rcv_buffer_acc = rcv_buffer.field_buf.get_host_access(h, sycl::write_only);
      h.host_task([=, &outbound_handle, &inbound_handle]() { // XXX pass handle arg here as copy, not ref!
        Net_PostSend(snd_buffer_acc.get_pointer(), message_len, NET_FLOAT64, rank_id, outbound_label, NET_GROUP_ALL, &outbound_handle);
        Net_PostReceive(rcv_buffer_acc.get_pointer(), message_len, NET_FLOAT64, rank_id, inbound_label, NET_GROUP_ALL, &inbound_handle);
      });
    });
  } else {
    globals.context.exec_ctx.submit([&](sycl::handler &h) {
      auto snd_buffer_acc = snd_buffer.field_buf.get_access<sycl::access_mode::read>(h);
      auto rcv_buffer_acc = rcv_buffer.field_buf.get_access<sycl::access_mode::write>(h);

      h.host_task([=, &outbound_handle, &inbound_handle](sycl::interop_handle ih) { // XXX pass handle arg here as copy, not ref!
        Net_PostSend(get_native_ptr_or_throw(ih, snd_buffer_acc), message_len, NET_FLOAT64, rank_id, outbound_label, NET_GROUP_ALL, &outbound_handle);
        Net_PostReceive(get_native_ptr_or_throw(ih, rcv_buffer_acc), message_len, NET_FLOAT64, rank_id, inbound_label, NET_GROUP_ALL, &inbound_handle);
      });
    });
  }
#else
  if (globals.settings.host_staging) {
    globals.context.exec_ctx.wait_and_throw();
    Net_PostSend(snd_buffer.access_ptr<R>(message_len), message_len, NET_FLOAT64, rank_id, outbound_label, NET_GROUP_ALL, &outbound_handle);
    Net_PostReceive(rcv_buffer.access_ptr<W>(message_len), message_len, NET_FLOAT64, rank_id, inbound_label, NET_GROUP_ALL, &inbound_handle);
  } else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    auto d = globals.context.exec_ctx.get_device();
    // Construct the buffers so copied get_pointer is not nullptr, only happens once per proc_index for the lifetime of the program
    if (!rcv_buffer.field_buf.get_pointer(d))
      globals.context.exec_ctx.submit([&](sycl::handler &h) { h.update(sycl::accessor{rcv_buffer.field_buf, h}); }).wait_and_throw();
    if (!snd_buffer.field_buf.get_pointer(d))
      globals.context.exec_ctx.submit([&](sycl::handler &h) { h.update(sycl::accessor{snd_buffer.field_buf, h}); }).wait_and_throw();
    // We can't use host_task here, but since we can pull out the pointers directly, if we synchronise before Net_AwaitEvery
    // the desired concurrency should still be there
    globals.context.exec_ctx.submit([&](sycl::handler &h) {
                           h.update(sycl::accessor{snd_buffer.field_buf, h, sycl::read_only});
                         })
        .wait_and_throw();
    globals.context.exec_ctx.submit([&](sycl::handler &h) {
                           h.update(sycl::accessor{rcv_buffer.field_buf, h, sycl::write_only});
                         })
        .wait_and_throw();
    Net_PostSend(snd_buffer.field_buf.get_pointer(d), message_len, NET_FLOAT64, rank_id, outbound_label, NET_GROUP_ALL, &outbound_handle);
    Net_PostReceive(rcv_buffer.field_buf.get_pointer(d), message_len, NET_FLOAT64, rank_id, inbound_label, NET_GROUP_ALL, &inbound_handle);
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}

void clover_wait_messages(sim_state &globals, int pending_requests, std::array<Net_Handle, 4> &msg_handle) {
  // need to make a call to wait / sync
#ifdef USE_HOSTTASK
  globals.context.exec_ctx
      .submit([&](sycl::handler &h) {
        h.host_task([=, &msg_handle]() { // XXX pass handle arg here as copy, not ref!
          Net_AwaitEvery(pending_requests, msg_handle.data(), NET_NO_STATUS);
        });
      })
      .wait_and_throw(); // don't proceed until Net_AwaitEvery call is done
#else
  if (globals.settings.host_staging) Net_AwaitEvery(pending_requests, msg_handle.data(), NET_NO_STATUS);
  else {
  #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
    globals.context.exec_ctx.wait_and_throw();
    Net_AwaitEvery(pending_requests, msg_handle.data(), NET_NO_STATUS);
  #else
    throw std::logic_error("host_task is disabled and staging is also disabled, this won't work");
  #endif
  }
#endif
}

void hydro2d_halo_swap(sim_state &globals, const int exchange_mask[FIELD_COUNT], const int halo_width) {

  // Assuming 1 patch per rank_id, this will be changed

  int horiz_halo_offset[FIELD_COUNT];
  int vert_halo_offset[FIELD_COUNT];

  std::array<Net_Handle, 4> msg_handle = {};
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

  //  std::cerr << "[C] LR = " << packed_len_horizontal << " BT = " << packed_len_vertical << std::endl;
  //  std::cerr << "[S] LR  = " << (globals.subdomain.left_rcv_buffer.field_buf.size() +
  //  globals.subdomain.right_rcv_buffer.field_buf.size()) << " BT = " <<  (globals.subdomain.top_rcv_buffer.field_buf.size() +
  //  globals.subdomain.bottom_rcv_buffer.field_buf.size()) << std::endl;

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

  clover_wait_messages(globals, pending_requests, msg_handle);

  // Copy back to the target_option
  //	Kokkos::deep_copy( left_rcv_buffer, hm_left_rcv_buffer);
  //	Kokkos::deep_copy( right_rcv_buffer, hm_right_rcv_buffer);

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

  clover_wait_messages(globals, pending_requests, msg_handle);

  // Copy back to the target_option
  //	Kokkos::deep_copy(globals.subdomain.bottom_rcv_buffer, hm_bottom_rcv_buffer);
  //	Kokkos::deep_copy(globals.subdomain.top_rcv_buffer, hm_top_rcv_buffer);

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
  // First copy send field_buf from target_option to host
  clover_send_recv_message(globals, subdomain_xlo, left_snd_buffer, left_rcv_buffer, message_len, outbound_label, inbound_label, outbound_handle, inbound_handle);
}
void hydro2d_post_msgs_right(sim_state &globals, hydro2d::DeviceArray1D<double> &right_snd_buffer,
                                    hydro2d::DeviceArray1D<double> &right_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                    Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  // First copy send field_buf from target_option to host
  clover_send_recv_message(globals, subdomain_xhi, right_snd_buffer, right_rcv_buffer, message_len, outbound_label, inbound_label, outbound_handle, inbound_handle);
}
void hydro2d_post_msgs_top(sim_state &globals, hydro2d::DeviceArray1D<double> &top_snd_buffer,
                                  hydro2d::DeviceArray1D<double> &top_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                  Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  // First copy send field_buf from target_option to host
  clover_send_recv_message(globals, subdomain_yhi, top_snd_buffer, top_rcv_buffer, message_len, outbound_label, inbound_label, outbound_handle, inbound_handle);
}
void hydro2d_post_msgs_bottom(sim_state &globals, hydro2d::DeviceArray1D<double> &bottom_snd_buffer,
                                     hydro2d::DeviceArray1D<double> &bottom_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                     Net_Handle &outbound_handle, Net_Handle &inbound_handle) {
  // First copy send field_buf from target_option to host
  clover_send_recv_message(globals, subdomain_ylo, bottom_snd_buffer, bottom_rcv_buffer, message_len, outbound_label, inbound_label, outbound_handle, inbound_handle);
}
