
#include <cstring>
#include <functional>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

#include "sim_state.h"
#include "field_dump.h"

namespace hydro2d {

std::ostream &operator<<(std::ostream &os, const Span1d &d) {
  os << "Span1d{"
     << " X[" << d.from << "->" << d.to << " (" << d.size << ")]"
     << "}";
  return os;
}
std::ostream &operator<<(std::ostream &os, const Span2d &d) {
  os << "Span2d{"
     << " X[" << d.startX << "->" << d.toX << " (" << d.lenX << ")]"
     << " Y[" << d.startY << "->" << d.toY << " (" << d.lenY << ")]"
     << "}";
  return os;
}
}


static void write_file(const std::string &label, const std::function<void(std::ofstream &)> &f) {
  std::ios_base::sync_with_stdio(false);
  std::ofstream out;
  out.open(label, std::ofstream::out | std::ofstream::trunc);
  std::vector<char> field_buf(1024 * 1024);
  out.rdbuf()->pubsetbuf(field_buf.data(), field_buf.size());
  f(out);
  out.close();
}

static void print_field(std::ostream &out, const std::string &label, hydro2d::DeviceArray1D<double> &field_buf) {
  auto host_vals = field_buf.host_copy();
  out << label << "(" << 1 << ") [" << field_buf.dim_len<0>() << "]\n";
  if (std::all_of(host_vals.begin(), host_vals.end(), [](auto x) { return x == 0.0; })) {
    out << "\t(0.0)";
  } else {
    out << "\t";
    for (double i : host_vals)
      out << i << ", ";
  }
  out << "\n";
}

static void print_field(std::ostream &out, const std::string &label, hydro2d::DeviceArray2D<double> &field_buf) {
  auto host_vals = field_buf.host_copy2d();
  out << label << "(" << 2 << ") [" << field_buf.dim_len<0>() << "x" << field_buf.dim_len<1>() << "]\n";
  out << "\t";
  if (std::all_of(host_vals.elems.begin(), host_vals.elems.end(), [](auto x) { return x == 0.0; })) {
    out << "\t(0.0)";
  } else {
    for (size_t i = 0; i < field_buf.dim_len<0>(); ++i) {
      for (size_t j = 0; j < field_buf.dim_len<1>(); ++j)
        out << host_vals(i,j) << ", ";
      out << "\t\n";
    }
  }
  out << "\n";
}

void hydro2d::write_snapshot(sim_state &g, const std::string &out_path) {
  if (g.settings.dumpPath.empty()) return;
  std::cout << "Dumping globals to " << out_path << std::endl;

  const auto dir = g.settings.dumpPath + "/";
  struct stat meta {};
  if (stat(dir.c_str(), &meta) != 0) {
    std::cout << "Creating " << dir << " for field write_snapshot" << std::endl;
    if (errno = 0; mkdir(dir.c_str(), 0777) != 0) {
      std::cerr << "Cannot create " << dir << ": " << std::strerror(errno) << ", skipping field write_snapshot" << std::endl;
      return;
    }
  } else if (meta.st_mode & S_IFDIR) {
  } else {
    std::cerr << dir << " already exists and is not a directory, skipping field write_snapshot" << std::endl;
    return;
  }

  write_file(dir + out_path, [&](std::ostream &out) {
    out << "Dump(tileCount = " << g.subdomain.tile_list.size() << ")\n";

    out << "error_status" << '=' << g.error_status << "\n";

    out << "cycle_index" << '=' << g.cycle_index << "\n";
    out << "remap_in_x" << '=' << g.remap_in_x << "\n";
    out << "time" << '=' << g.time << "\n";

    out << "dt" << '=' << g.dt << "\n";
    out << "dt_prev" << '=' << g.dt_prev << "\n";

    out << "run_finished" << '=' << g.run_finished << "\n";
    out << "jdt" << '=' << g.jdt << "\n";
    out << "kdt" << '=' << g.kdt << "\n";

    for (size_t i = 0; i < g.subdomain.tile_list.size(); ++i) {
      auto &fs = g.subdomain.tile_list[i].field;
      out << "\tTile[ " << i << "]:\n";

      tile_desc &meta = g.subdomain.tile_list[i].meta;
      for (int l = 0; l < 4; ++l) {
        out << "meta.tile_adjacent[i]" << '=' << meta.tile_adjacent[i] << "\n";
        out << "meta.edge_mask[i]" << '=' << meta.edge_mask[i] << "\n";
      }

      out << "meta.tile_xlo" << '=' << meta.tile_xlo << "\n";
      out << "meta.blk_xmax" << '=' << meta.blk_xmax << "\n";
      out << "meta.blk_ymin" << '=' << meta.blk_ymin << "\n";
      out << "meta.blk_ymax" << '=' << meta.blk_ymax << "\n";
      out << "meta.tile_west_idx" << '=' << meta.tile_west_idx << "\n";
      out << "meta.right_idx" << '=' << meta.right_idx << "\n";
      out << "meta.blk_bottom" << '=' << meta.blk_bottom << "\n";
      out << "meta.blk_top" << '=' << meta.blk_top << "\n";

      print_field(out, "density0", fs.density0);
      print_field(out, "density1", fs.density1);
      print_field(out, "energy0", fs.energy0);
      print_field(out, "spec_energy1", fs.spec_energy1);
      print_field(out, "cell_pressure", fs.cell_pressure);
      print_field(out, "shock_damping", fs.shock_damping);
      print_field(out, "wave_speed", fs.wave_speed);
      print_field(out, "velx0", fs.velx0);
      print_field(out, "velx1", fs.velx1);
      print_field(out, "vely0", fs.vely0);
      print_field(out, "vely1", fs.vely1);
      print_field(out, "volume_flux_x", fs.volume_flux_x);
      print_field(out, "volume_flux_y", fs.volume_flux_y);
      print_field(out, "face_mass_flux_x", fs.face_mass_flux_x);
      print_field(out, "face_mass_flux_y", fs.face_mass_flux_y);

      print_field(out, "scratch_buf1", fs.scratch_buf1);
      print_field(out, "scratch_buf2", fs.scratch_buf2);
      print_field(out, "scratch_buf3", fs.scratch_buf3);
      print_field(out, "scratch_buf4", fs.scratch_buf4);
      print_field(out, "scratch_buf5", fs.scratch_buf5);
      print_field(out, "scratch_buf6", fs.scratch_buf6);
      print_field(out, "scratch_buf7", fs.scratch_buf7);

      print_field(out, "xcentre", fs.xcentre);
      print_field(out, "dxcell", fs.dxcell);
      print_field(out, "ycentre", fs.ycentre);
      print_field(out, "dycell", fs.dycell);
      print_field(out, "xnode", fs.xnode);
      print_field(out, "dxnode", fs.dxnode);
      print_field(out, "ynode", fs.ynode);
      print_field(out, "dynode", fs.dynode);

      print_field(out, "cell_volume", fs.cell_volume);
      print_field(out, "xface_area", fs.xface_area);
      print_field(out, "yface_area", fs.yface_area);
      out << std::flush;
    }
  });
}
