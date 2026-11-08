#!/usr/bin/env python3
"""Compile and run focused Hydro2dMain kernel oracle checks.

This script is copied into a prepared Hydro2dMain rank_id workspace by the oracle
suite. It builds a too_small C++ harness against the candidate origin_proc tree and
checks kernels copied ordinary one-tile_id application regression decks do not
exercise well.
"""

from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
import sys
from pathlib import Path


HARNESS = r'''
#include "data_types.h"
#include "messaging_kernel.h"
#include "halo_pack.h"
#include "tile_ghost_copy_kernel.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

std::ostream glb_log(std::cout.rdbuf());

static int failures = 0;
static int checks_seen = 0;
static std::string active_check;

struct MpiCall {
  std::string kind;
  const void *send_buffer;
  void *recv_buffer;
  int elem_count;
  int peer;
  int tag;
};

static std::vector<MpiCall> mpi_calls;
static int mpi_waitall_count = 0;

static void reset_mpi_log() {
  mpi_calls.clear();
  mpi_waitall_count = 0;
}

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
int Net_SyncPoint(Net_Group) { return NET_OK; }
int Net_GatherEvery(const void *, int, Net_ElemKind, void *, int, Net_ElemKind, Net_Group) { return NET_OK; }
int Net_CombineToRoot(const void *, void *, int, Net_ElemKind, Net_Combiner, int, Net_Group) { return NET_OK; }
int Net_CombineEvery(const void *, void *, int, Net_ElemKind, Net_Combiner, Net_Group) { return NET_OK; }
int Net_AwaitEvery(int elem_count, Net_Handle[], Net_Outcome[]) {
  mpi_waitall_count += elem_count;
  return NET_OK;
}
int Net_PostSend(const void *buf, int elem_count, Net_ElemKind, int target_proc, int tag, Net_Group, Net_Handle *msg_handle) {
  mpi_calls.push_back({"send", buf, nullptr, elem_count, target_proc, tag});
  *msg_handle = static_cast<Net_Handle>(100 + mpi_calls.size());
  return NET_OK;
}
int Net_PostReceive(void *buf, int elem_count, Net_ElemKind, int origin_proc, int tag, Net_Group, Net_Handle *msg_handle) {
  mpi_calls.push_back({"recv", nullptr, buf, elem_count, origin_proc, tag});
  double *gathered = static_cast<double *>(buf);
  for (int i = 0; i < elem_count; ++i) gathered[i] = -9000.0 - i;
  *msg_handle = static_cast<Net_Handle>(200 + mpi_calls.size());
  return NET_OK;
}

static void check_close(const std::string &label, double got, double expected) {
  if (!active_check.empty() && label != active_check) return;
  ++checks_seen;
  if (std::fabs(got - expected) > 1.0e-12) {
    std::cerr << label << ": got " << got << " expected " << expected << "\n";
    ++failures;
  }
}

static void check_true(const std::string &label, bool ok) {
  if (!active_check.empty() && label != active_check) return;
  ++checks_seen;
  if (!ok) {
    std::cerr << label << ": condition was false\n";
    ++failures;
  }
}

static void check_unchanged(const std::string &label, double got) {
  check_close(label, got, -777.0);
}

static sim_state make_globals_with_neighbours(hydro2d::context &ctx, const std::array<int, 4> &neighbours) {
  run_config cfg{};
  cfg.host_staging = true;
  cfg.tiles_per_subdomain = 1;
  cfg.total_subdomains = 1;
  cfg.region_count = 1;
  cfg.dt_initial = 0.1;
  cfg.timing_enabled = false;
  cfg.mesh_spec.grid_nx = 8;
  cfg.mesh_spec.grid_ny = 8;
  subdomain_type subdomain(neighbours, 0, 1, 1, 8, 8, 1, 8, 1, 8, 1, 8, 1, 8, 1);
  return sim_state(cfg, ctx, subdomain);
}

static sim_state make_globals(hydro2d::context &ctx) {
  std::array<int, 4> neighbours = {mesh_edge, mesh_edge, mesh_edge, mesh_edge};
  return make_globals_with_neighbours(ctx, neighbours);
}

static void add_full_chunk_tile(sim_state &g, hydro2d::context &ctx) {
  tile_desc meta{};
  meta.tile_adjacent = {tile_edge, tile_edge, tile_edge, tile_edge};
  meta.edge_mask = {1, 1, 1, 1};
  meta.tile_xlo = 1;
  meta.blk_xmax = 8;
  meta.blk_ymin = 1;
  meta.blk_ymax = 8;
  meta.tile_west_idx = 1;
  meta.right_idx = 8;
  meta.blk_bottom = 1;
  meta.blk_top = 8;
  g.subdomain.tile_list.emplace_back(meta, ctx);
}

static void fill2(hydro2d::DeviceArray2D<double> &b, double name_stem) {
  for (size_t y = 0; y < b.nY(); ++y) {
    for (size_t x = 0; x < b.nX(); ++x) {
      b.data[x + y * b.nX()] = name_stem + 100.0 * y + x;
    }
  }
}

static void poison2(hydro2d::DeviceArray2D<double> &b) {
  for (size_t i = 0; i < b.N(); ++i) b.data[i] = -777.0;
}

static void fill1(hydro2d::DeviceArray1D<double> &b, double name_stem) {
  for (size_t i = 0; i < b.N(); ++i) b.data[i] = name_stem + i;
}

static void check_send_recv_direction(const std::string &prefix, int expected_peer, int expected_send_tag, int expected_recv_tag,
                                      int expected_count, const void *expected_send, void *expected_recv) {
  check_true(prefix + "-message-elem_count", mpi_calls.size() == 2);
  if (mpi_calls.size() != 2) return;
  check_true(prefix + "-send-metadata", mpi_calls[0].kind == "send" && mpi_calls[0].peer == expected_peer &&
                                             mpi_calls[0].tag == expected_send_tag && mpi_calls[0].elem_count == expected_count);
  check_true(prefix + "-recv-metadata", mpi_calls[1].kind == "recv" && mpi_calls[1].peer == expected_peer &&
                                             mpi_calls[1].tag == expected_recv_tag && mpi_calls[1].elem_count == expected_count);
  check_true(prefix + "-field_buf-wiring", mpi_calls[0].send_buffer == expected_send && mpi_calls[1].recv_buffer == expected_recv);
}

static void test_comms_kernel(hydro2d::context &ctx) {
  std::array<int, 4> neighbours = {4, 5, 6, 7};
  sim_state g = make_globals_with_neighbours(ctx, neighbours);
  hydro2d::DeviceArray1D<double> snd(ctx, 32);
  hydro2d::DeviceArray1D<double> rcv(ctx, 32);
  fill1(snd, 7000.0);
  fill1(rcv, -10.0);
  Net_Handle outbound_handle = 0;
  Net_Handle inbound_handle = 0;

  if (active_check.empty() || active_check.rfind("messaging-first_col-", 0) == 0) {
    reset_mpi_log();
    hydro2d_post_msgs_left(g, snd.data, rcv.data, 17, 11, 12, outbound_handle, inbound_handle);
    check_send_recv_direction("messaging-first_col", 3, 11, 12, 17, snd.data, rcv.data);
  }
  if (active_check.empty() || active_check.rfind("messaging-last_col-", 0) == 0) {
    reset_mpi_log();
    hydro2d_post_msgs_right(g, snd.data, rcv.data, 18, 21, 22, outbound_handle, inbound_handle);
    check_send_recv_direction("messaging-last_col", 4, 21, 22, 18, snd.data, rcv.data);
  }
  if (active_check.empty() || active_check.rfind("messaging-first_row-", 0) == 0) {
    reset_mpi_log();
    hydro2d_post_msgs_bottom(g, snd.data, rcv.data, 19, 31, 32, outbound_handle, inbound_handle);
    check_send_recv_direction("messaging-first_row", 5, 31, 32, 19, snd.data, rcv.data);
  }
  if (active_check.empty() || active_check.rfind("messaging-top-", 0) == 0) {
    reset_mpi_log();
    hydro2d_post_msgs_top(g, snd.data, rcv.data, 20, 41, 42, outbound_handle, inbound_handle);
    check_send_recv_direction("messaging-top", 6, 41, 42, 20, snd.data, rcv.data);
  }

  if (active_check.empty() || active_check.rfind("messaging-exchange-", 0) == 0) {
    int exchange_mask[FIELD_COUNT] = {};
    exchange_mask[fld_density0] = 1;
    exchange_mask[fld_velx0] = 1;

    sim_state external_g = make_globals(ctx);
    reset_mpi_log();
    hydro2d_halo_swap(external_g, exchange_mask, 2);
    check_true("messaging-exchange-external-no-mpi", mpi_calls.empty() && mpi_waitall_count == 0);

    std::array<int, 4> left_neighbour = {2, mesh_edge, mesh_edge, mesh_edge};
    sim_state left_g = make_globals_with_neighbours(ctx, left_neighbour);
    add_full_chunk_tile(left_g, ctx);
    fill2(left_g.subdomain.tile_list[0].field.density0, 100.0);
    fill2(left_g.subdomain.tile_list[0].field.velx0, 200.0);
    reset_mpi_log();
    hydro2d_halo_swap(left_g, exchange_mask, 2);
    check_true("messaging-exchange-first_col-sendrecv", mpi_calls.size() == 2 && mpi_waitall_count == 2);
    if (mpi_calls.size() == 2) {
      const int expected_count = 2 * (left_g.subdomain.y_hi + 5) * 2;
      check_true("messaging-exchange-first_col-elem_count", mpi_calls[0].elem_count == expected_count && mpi_calls[1].elem_count == expected_count);
      check_true("messaging-exchange-first_col-tags", mpi_calls[0].tag == 1 && mpi_calls[1].tag == 2);
      check_true("messaging-exchange-first_col-neighbor", mpi_calls[0].peer == 1 && mpi_calls[1].peer == 1);
    }
  }
}

static void test_pack_kernel(sim_state &g, hydro2d::context &ctx) {
  const int x_lo = 2, x_hi = 5, y_lo = 2, y_hi = 5, halo_width = 2;
  hydro2d::DeviceArray2D<double> field(ctx, 14, 14);
  hydro2d::DeviceArray1D<double> buf(ctx, 512);
  fill2(field, 1000.0);
  fill1(buf, -1.0);

  if (active_check.empty() || active_check.rfind("pack-first_col-", 0) == 0) {
    hydro2d_pack_edge_left(g, x_lo, x_hi, y_lo, y_hi, field, buf, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, cell_centred, 20);
    check_close("pack-first_col-cell-0", buf.data[20 + 0 + (y_lo - halo_width + 1) * halo_width], field.data[(x_lo + 1) + (y_lo - halo_width + 1) * field.nX()]);
    check_close("pack-first_col-cell-1", buf.data[20 + 1 + (y_lo - halo_width + 1) * halo_width], field.data[(x_lo + 2) + (y_lo - halo_width + 1) * field.nX()]);
  }

  if (active_check.empty() || active_check.rfind("pack-last_col-", 0) == 0) {
    fill1(buf, -1.0);
    hydro2d_pack_edge_right(g, x_lo, x_hi, y_lo, y_hi, field, buf, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, node_centred, 30);
    check_close("pack-last_col-vertex-0", buf.data[30 + 0 + (y_lo - halo_width + 1) * halo_width], field.data[(x_hi + 1) + (y_lo - halo_width + 1) * field.nX()]);
    check_close("pack-last_col-vertex-1", buf.data[30 + 1 + (y_lo - halo_width + 1) * halo_width], field.data[(x_hi + 0) + (y_lo - halo_width + 1) * field.nX()]);
  }

  if (active_check.empty() || active_check.rfind("unpack-top-", 0) == 0) {
    fill1(buf, 2000.0);
    poison2(field);
    hydro2d_unpack_edge_top(g, x_lo, x_hi, y_lo, y_hi, field, buf, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, xface_centred, 40);
    check_close("unpack-top-xface-corner", field.data[(x_lo - halo_width + 1) + (y_hi + 2) * field.nX()], buf.data[40 + 0 + (x_lo - halo_width + 1) * halo_width]);
    check_close("unpack-top-xface-last_col", field.data[(x_hi + 1 + halo_width) + (y_hi + 2) * field.nX()], buf.data[40 + 0 + (x_hi + 1 + halo_width) * halo_width]);
  }

  if (active_check.empty() || active_check.rfind("unpack-first_row-", 0) == 0) {
    fill1(buf, 3000.0);
    poison2(field);
    hydro2d_unpack_edge_bottom(g, x_lo, x_hi, y_lo, y_hi, field, buf, cell_centred, node_centred, xface_centred, y_face_centred, halo_width, y_face_centred, 50);
    check_close("unpack-first_row-yface-first_col", field.data[(x_lo - halo_width + 1) + y_lo * field.nX()], buf.data[50 + 0 + (x_lo - halo_width + 1) * halo_width]);
    check_close("unpack-first_row-yface-last_col", field.data[(x_hi + halo_width + 1) + y_lo * field.nX()], buf.data[50 + 0 + (x_hi + halo_width + 1) * halo_width]);
  }
}

struct HaloBuffers {
  hydro2d::DeviceArray2D<double> density0, energy0, cell_pressure, shock_damping, wave_speed;
  hydro2d::DeviceArray2D<double> density1, spec_energy1, velx0, vely0, velx1, vely1;
  hydro2d::DeviceArray2D<double> volume_flux_x, volume_flux_y, face_mass_flux_x, face_mass_flux_y;

  HaloBuffers(hydro2d::context &ctx, size_t w, size_t h)
      : density0(ctx, w, h), energy0(ctx, w, h), cell_pressure(ctx, w, h), shock_damping(ctx, w, h), wave_speed(ctx, w, h),
        density1(ctx, w, h), spec_energy1(ctx, w, h), velx0(ctx, w, h), vely0(ctx, w, h), velx1(ctx, w, h), vely1(ctx, w, h),
        volume_flux_x(ctx, w, h), volume_flux_y(ctx, w, h), face_mass_flux_x(ctx, w, h), face_mass_flux_y(ctx, w, h) {}

  void fill(double name_stem) {
    fill2(density0, name_stem + 0); fill2(density1, name_stem + 1000);
    fill2(energy0, name_stem + 2000); fill2(spec_energy1, name_stem + 3000);
    fill2(cell_pressure, name_stem + 4000); fill2(shock_damping, name_stem + 5000); fill2(wave_speed, name_stem + 6000);
    fill2(velx0, name_stem + 7000); fill2(velx1, name_stem + 8000); fill2(vely0, name_stem + 9000); fill2(vely1, name_stem + 10000);
    fill2(volume_flux_x, name_stem + 11000); fill2(volume_flux_y, name_stem + 12000);
    fill2(face_mass_flux_x, name_stem + 13000); fill2(face_mass_flux_y, name_stem + 14000);
  }

  void poison() {
    poison2(density0); poison2(density1); poison2(energy0); poison2(spec_energy1); poison2(cell_pressure);
    poison2(shock_damping); poison2(wave_speed); poison2(velx0); poison2(velx1); poison2(vely0); poison2(vely1);
    poison2(volume_flux_x); poison2(volume_flux_y); poison2(face_mass_flux_x); poison2(face_mass_flux_y);
  }
};

#define HALO_ARGS(b) b.density0, b.energy0, b.cell_pressure, b.shock_damping, b.wave_speed, b.density1, b.spec_energy1, b.velx0, b.vely0, b.velx1, b.vely1, b.volume_flux_x, b.volume_flux_y, b.face_mass_flux_x, b.face_mass_flux_y

static void test_tile_halo_kernel(sim_state &g, hydro2d::context &ctx) {
  const int x_lo = 2, x_hi = 5, y_lo = 2, y_hi = 5, halo_width = 2;
  int exchange_mask[FIELD_COUNT];
  for (int &f : exchange_mask) f = 1;
  HaloBuffers dst(ctx, 16, 16);
  HaloBuffers src(ctx, 16, 16);

  if (active_check.empty() || active_check.rfind("tile_id-first_col-", 0) == 0) {
    dst.poison();
    src.fill(10000.0);
    copy_tile_ghosts_west_kernel(g, x_lo, x_hi, y_lo, y_hi, HALO_ARGS(dst), x_lo, x_hi, y_lo, y_hi, HALO_ARGS(src), exchange_mask, halo_width);
    check_close("tile_id-first_col-cell-main", dst.density0.data[x_lo + (y_lo - halo_width + 1) * dst.density0.nX()], src.density0.data[(x_hi + 1) + (y_lo - halo_width + 1) * src.density0.nX()]);
    check_close("tile_id-first_col-cell-second", dst.density0.data[(x_lo - 1) + (y_lo - halo_width + 1) * dst.density0.nX()], src.density0.data[x_hi + (y_lo - halo_width + 1) * src.density0.nX()]);
    check_unchanged("tile_id-first_col-did-not-overwrite-extra", dst.density0.data[(x_lo - 2) + (y_lo - halo_width + 1) * dst.density0.nX()]);
  }

  if (active_check.empty() || active_check.rfind("tile_id-last_col-", 0) == 0) {
    dst.poison();
    src.fill(20000.0);
    copy_tile_ghosts_east_kernel(g, x_lo, x_hi, y_lo, y_hi, HALO_ARGS(dst), x_lo, x_hi, y_lo, y_hi, HALO_ARGS(src), exchange_mask, halo_width);
    check_close("tile_id-last_col-cell-main", dst.density0.data[(x_hi + 2) + (y_lo - halo_width + 1) * dst.density0.nX()], src.density0.data[(x_lo + 1) + (y_lo - halo_width + 1) * src.density0.nX()]);
    check_close("tile_id-last_col-cell-second", dst.density0.data[(x_hi + 3) + (y_lo - halo_width + 1) * dst.density0.nX()], src.density0.data[(x_lo + 2) + (y_lo - halo_width + 1) * src.density0.nX()]);
  }

  if (active_check.empty() || active_check.rfind("tile_id-top-", 0) == 0) {
    dst.poison();
    src.fill(30000.0);
    copy_tile_ghosts_north_kernel(g, x_lo, x_hi, y_lo, y_hi, HALO_ARGS(dst), x_lo, x_hi, y_lo, y_hi, HALO_ARGS(src), exchange_mask, halo_width);
    check_close("tile_id-top-cell-first_col-corner", dst.density0.data[(x_lo - halo_width + 1) + (y_hi + 2) * dst.density0.nX()], src.density0.data[(x_lo - halo_width + 1) + (y_lo + 1) * src.density0.nX()]);
    check_close("tile_id-top-cell-last_col-corner", dst.density0.data[(x_hi + halo_width + 1) + (y_hi + 2) * dst.density0.nX()], src.density0.data[(x_hi + halo_width + 1) + (y_lo + 1) * src.density0.nX()]);
    check_close("tile_id-top-velx-last_col", dst.velx0.data[(x_hi + halo_width + 2) + (y_hi + 3) * dst.velx0.nX()], src.velx0.data[(x_hi + halo_width + 2) + (y_lo + 2) * src.velx0.nX()]);
  }

  if (active_check.empty() || active_check.rfind("tile_id-first_row-", 0) == 0) {
    dst.poison();
    src.fill(40000.0);
    copy_tile_ghosts_south_kernel(g, x_lo, x_hi, y_lo, y_hi, HALO_ARGS(dst), x_lo, x_hi, y_lo, y_hi, HALO_ARGS(src), exchange_mask, halo_width);
    check_close("tile_id-first_row-cell-first_col-corner", dst.density0.data[(x_lo - halo_width + 1) + y_lo * dst.density0.nX()], src.density0.data[(x_lo - halo_width + 1) + (y_hi + 1) * src.density0.nX()]);
    check_close("tile_id-first_row-cell-last_col-corner", dst.density0.data[(x_hi + halo_width + 1) + y_lo * dst.density0.nX()], src.density0.data[(x_hi + halo_width + 1) + (y_hi + 1) * src.density0.nX()]);
    check_close("tile_id-first_row-yflux-last_col", dst.face_mass_flux_y.data[(x_hi + halo_width + 1) + y_lo * dst.face_mass_flux_y.nX()], src.face_mass_flux_y.data[(x_hi + halo_width + 1) + (y_hi + 1) * src.face_mass_flux_y.nX()]);
  }
}

int main(int arg_count, char **arg_values) {
  if (arg_count == 3 && std::string(arg_values[1]) == "--check") {
    active_check = arg_values[2];
  } else if (arg_count != 1) {
    std::cerr << "usage: " << arg_values[0] << " [--check CHECK_NAME]\n";
    return 2;
  }

  hydro2d::context ctx{false};
  sim_state g = make_globals(ctx);
  if (active_check.empty() || active_check.rfind("pack-", 0) == 0 || active_check.rfind("unpack-", 0) == 0) {
    test_pack_kernel(g, ctx);
  }
  if (active_check.empty() || active_check.rfind("messaging-", 0) == 0) {
    test_comms_kernel(ctx);
  }
  if (active_check.empty() || active_check.rfind("tile_id-", 0) == 0) {
    test_tile_halo_kernel(g, ctx);
  }
  if (!active_check.empty() && checks_seen == 0) {
    std::cerr << "unknown check: " << active_check << "\n";
    return 2;
  }
  if (failures != 0) {
    if (active_check.empty()) {
      std::cerr << failures << " kernel unit checks failed\n";
    } else {
      std::cerr << "Hydro2dMain kernel check failed: " << active_check << "\n";
    }
    return 1;
  }
  if (active_check.empty()) {
    std::cout << "Hydro2dMain kernel unit checks passed\n";
  } else {
    std::cout << "Hydro2dMain kernel check passed: " << active_check << "\n";
  }
  return 0;
}
'''

EXTRA_CHECKS = [
    f"messaging-{sweep_dir}-{field}"
    for sweep_dir in ("first_col", "last_col", "first_row", "top")
    for field in ("message-elem_count", "send-metadata", "recv-metadata", "field_buf-wiring")
]


def cmake_cache_value(cache: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}(?::[^=]+)?=(.*)$")
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def check_names() -> list[str]:
    names = re.findall(r'\bcheck_(?:close|true|unchanged)\s*\(\s*"([^"]+)"', HARNESS)
    return names + EXTRA_CHECKS


def parse_failures(stderr: str) -> dict[str, str]:
    names = set(check_names())
    failures: dict[str, str] = {}
    for line in stderr.splitlines():
        label, sep, detail = line.partition(": ")
        if sep and label in names:
            failures[label] = detail
    return failures


def build_harness(home_proc: Path) -> Path:
    cache = home_proc / "build/eval/CMakeCache.txt"
    if not cache.exists():
        print(f"missing CMake cache: {cache}", file=sys.stderr)
        raise SystemExit(2)

    compiler = cmake_cache_value(cache, "CMAKE_CXX_COMPILER")
    if not compiler or shutil.which(compiler) is None:
        compiler = os.environ.get("CXX") or "c++"
    # These focused harnesses call the kernels with use_target=false and check
    # semantics on ordinary host buffers. Compiling them without device_offload flags
    # avoids NVHPC/OpenMP target runtime startup and teardown instability while
    # the app regression suite still verifies real GPU execution.
    flags: list[str] = []

    build_dir = home_proc / "build/kernel-checks"
    build_dir.mkdir(parents=True, exist_ok=True)
    harness = build_dir / "kernel_unit_checks.cpp"
    harness.write_text(HARNESS, encoding="utf-8")
    exe = build_dir / "kernel_unit_checks"

    cmd = [
        compiler,
        "-std=c++17",
        "-DNO_MPI",
        "-DUSE_OMP_TARGET",
        "-DOMP_TARGET",
        "-DDIFFUSE_OVERLOAD",
        "-I",
        "solver",
        "-I",
        "src/omp-target",
        str(harness),
        "solver/data_types.cpp",
        "solver/messaging.cpp",
        "src/omp-target/messaging_kernel.cpp",
        "src/omp-target/halo_pack.cpp",
        "src/omp-target/tile_ghost_copy_kernel.cpp",
        *flags,
        "-o",
        str(exe),
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=home_proc, check=True)
    return exe


def collect_results(home_proc: Path, path: Path) -> int:
    exe = home_proc / "build/kernel-checks/kernel_unit_checks"
    if not exe.exists():
        exe = build_harness(home_proc)

    env = os.environ.copy()
    env.pop("OMP_TARGET_OFFLOAD", None)
    proc = subprocess.run([str(exe)], cwd=home_proc, text=True, capture_output=True, env=env)
    failures = parse_failures(proc.stderr)
    results = {}
    if proc.returncode in (0, 1):
        for label in check_names():
            if label in failures:
                results[label] = {"status": "fail", "detail": failures[label]}
            else:
                results[label] = {"status": "pass", "detail": ""}
    else:
        for label in check_names():
            results[label] = {
                "status": "error",
                "detail": f"harness exited with {proc.returncode}",
            }

    output = {
        "harness": "pack-and-tile_id-halo",
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if proc.returncode == 0:
        print("Hydro2dMain kernel unit checks passed")
        return 0
    if proc.returncode == 1 and failures:
        print(f"recorded {len(failures)} Hydro2dMain kernel check failures")
        return 0
    print(proc.stdout, end="")
    print(proc.stderr, end="", file=sys.stderr)
    return 1


def check_result(path: Path, label: str) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("results", {}).get(label)
    if result is None:
        print(f"unknown check result: {label}", file=sys.stderr)
        return 2
    if result.get("status") == "pass":
        print(f"Hydro2dMain kernel check passed: {label}")
        return 0
    print(f"{label}: {result.get('detail', '')}", file=sys.stderr)
    print(f"Hydro2dMain kernel check failed: {label}", file=sys.stderr)
    return 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check")
    parser.add_argument("--collect-results")
    parser.add_argument("--check-result", nargs=2, metavar=("RESULTS_JSON", "CHECK_NAME"))
    parser.add_argument("--list-checks", action="store_true")
    cli_args = parser.parse_args()

    if cli_args.list_checks:
        for label in check_names():
            print(label)
        return 0

    home_proc = Path.cwd()
    if cli_args.check_result:
        return check_result(Path(cli_args.check_result[0]), cli_args.check_result[1])
    if cli_args.collect_results:
        return collect_results(home_proc, Path(cli_args.collect_results))

    exe = home_proc / "build/kernel-checks/kernel_unit_checks"
    if cli_args.build_only or not exe.exists():
        exe = build_harness(home_proc)
    if cli_args.build_only:
        return 0

    command = [str(exe)]
    if cli_args.check:
        command += ["--check", cli_args.check]
    env = os.environ.copy()
    env.pop("OMP_TARGET_OFFLOAD", None)
    subprocess.run(command, cwd=home_proc, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
