#pragma once

#include "rank_comm.h"
#include "device_context.h"
#include "sim_state.h"

enum data_centring { cell_centred = 1, node_centred = 2, xface_centred = 3, y_face_centred = 4 };

void hydro2d_post_msgs_left(sim_state &globals, hydro2d::TransferPtr1D<double> left_snd_buffer,
                                   hydro2d::TransferPtr1D<double> left_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                   Net_Handle &outbound_handle, Net_Handle &inbound_handle);

void hydro2d_post_msgs_right(sim_state &globals, hydro2d::TransferPtr1D<double> right_snd_buffer,
                                    hydro2d::TransferPtr1D<double> right_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                    Net_Handle &outbound_handle, Net_Handle &inbound_handle);
void hydro2d_post_msgs_bottom(sim_state &globals, hydro2d::TransferPtr1D<double> bottom_snd_buffer,
                                     hydro2d::TransferPtr1D<double> top_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                     Net_Handle &outbound_handle, Net_Handle &inbound_handle);
void hydro2d_post_msgs_top(sim_state &globals, hydro2d::TransferPtr1D<double> top_snd_buffer,
                                  hydro2d::TransferPtr1D<double> top_rcv_buffer, int message_len, int outbound_label, int inbound_label,
                                  Net_Handle &outbound_handle, Net_Handle &inbound_handle);
void hydro2d_halo_swap(sim_state &globals, const int exchange_mask[FIELD_COUNT], int halo_width);

void hydro2d_pack_xlo(sim_state &globals, hydro2d::DeviceArray1D<double> &, int tile_id, const int exchange_mask[FIELD_COUNT], int halo_width,
                      int horiz_halo_offset[FIELD_COUNT]);

void hydro2d_pack_xhi(sim_state &globals, hydro2d::DeviceArray1D<double> &, int tile_id, const int exchange_mask[FIELD_COUNT], int halo_width,
                       int horiz_halo_offset[FIELD_COUNT]);

void hydro2d_pack_ylo(sim_state &globals, hydro2d::DeviceArray1D<double> &, int tile_id, const int exchange_mask[FIELD_COUNT], int halo_width,
                        int vert_halo_offset[FIELD_COUNT]);

void hydro2d_pack_yhi(sim_state &globals, hydro2d::DeviceArray1D<double> &, int tile_id, const int exchange_mask[FIELD_COUNT], int halo_width,
                     int vert_halo_offset[FIELD_COUNT]);

void hydro2d_unpack_xlo(sim_state &globals, hydro2d::DeviceArray1D<double> &, const int exchange_mask[FIELD_COUNT], int tile_id, int halo_width,
                        int horiz_halo_offset[FIELD_COUNT]);

void hydro2d_unpack_xhi(sim_state &globals, hydro2d::DeviceArray1D<double> &, const int exchange_mask[FIELD_COUNT], int tile_id, int halo_width,
                         int horiz_halo_offset[FIELD_COUNT]);

void hydro2d_unpack_ylo(sim_state &globals, hydro2d::DeviceArray1D<double> &, const int exchange_mask[FIELD_COUNT], int tile_id, int halo_width,
                          int vert_halo_offset[FIELD_COUNT]);

void hydro2d_unpack_yhi(sim_state &globals, hydro2d::DeviceArray1D<double> &, const int exchange_mask[FIELD_COUNT], int tile_id, int halo_width,
                       int vert_halo_offset[FIELD_COUNT]);

void hydro2d_pack_edge_left(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                              hydro2d::DeviceArray1D<double> &left_snd_buffer, int cell_centred, int node_centred, int xface_kind_arg, int y_face_centred,
                              int halo_width, int field_set, int slot_start);
void hydro2d_pack_edge_right(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                               hydro2d::DeviceArray1D<double> &right_snd_buffer, int cell_centred, int node_centred, int xface_kind_arg, int y_face_centred,
                               int halo_width, int field_set, int slot_start);
void hydro2d_unpack_edge_left(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                hydro2d::DeviceArray1D<double> &left_rcv_buffer, int cell_centred, int node_centred, int xface_kind_arg, int y_face_centred,
                                int halo_width, int field_set, int slot_start);
void hydro2d_unpack_edge_right(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                 hydro2d::DeviceArray1D<double> &right_rcv_buffer, int cell_centred, int node_centred, int xface_kind_arg,
                                 int y_face_centred, int halo_width, int field_set, int slot_start);
void hydro2d_pack_edge_bottom(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                hydro2d::DeviceArray1D<double> &bottom_snd_buffer, int cell_centred, int node_centred, int xface_kind_arg,
                                int y_face_centred, int halo_width, int field_set, int slot_start);
void hydro2d_pack_edge_top(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                             hydro2d::DeviceArray1D<double> &top_snd_buffer, int cell_centred, int node_centred, int xface_kind_arg, int y_face_centred,
                             int halo_width, int field_set, int slot_start);
void hydro2d_unpack_edge_bottom(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                                  hydro2d::DeviceArray1D<double> &bottom_rcv_buffer, int cell_centred, int node_centred, int xface_kind_arg,
                                  int y_face_centred, int halo_width, int field_set, int slot_start);
void hydro2d_unpack_edge_top(sim_state &sim, int x_lo, int x_hi, int y_lo, int y_hi, hydro2d::DeviceArray2D<double> &field,
                               hydro2d::DeviceArray1D<double> &top_rcv_buffer, int cell_centred, int node_centred, int xface_kind_arg, int y_face_centred,
                               int halo_width, int field_set, int slot_start);
