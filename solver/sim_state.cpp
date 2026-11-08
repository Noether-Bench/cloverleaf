
#include "sim_state.h"

field_set::field_set(const size_t x_extent, const size_t y_extent, hydro2d::context &ctx)
    : density0(ctx, x_extent, y_extent),
      density1(ctx, x_extent, y_extent),
      energy0(ctx, x_extent, y_extent),
      spec_energy1(ctx, x_extent, y_extent),
      cell_pressure(ctx, x_extent, y_extent),
      shock_damping(ctx, x_extent, y_extent),
      wave_speed(ctx, x_extent, y_extent),
      velx0(ctx, x_extent + 1, y_extent + 1),
      velx1(ctx, x_extent + 1, y_extent + 1),
      vely0(ctx, x_extent + 1, y_extent + 1),
      vely1(ctx, x_extent + 1, y_extent + 1),
      volume_flux_x(ctx, x_extent + 1, y_extent),
      face_mass_flux_x(ctx, x_extent + 1, y_extent),
      volume_flux_y(ctx, x_extent, y_extent + 1),
      face_mass_flux_y(ctx, x_extent, y_extent + 1),
      scratch_buf1(ctx, x_extent + 1, y_extent + 1),
      scratch_buf2(ctx, x_extent + 1, y_extent + 1),
      scratch_buf3(ctx, x_extent + 1, y_extent + 1),
      scratch_buf4(ctx, x_extent + 1, y_extent + 1),
      scratch_buf5(ctx, x_extent + 1, y_extent + 1),
      scratch_buf6(ctx, x_extent + 1, y_extent + 1),
      scratch_buf7(ctx, x_extent + 1, y_extent + 1),
      xcentre(ctx, x_extent),
      dxcell(ctx, x_extent),
      ycentre(ctx, y_extent),
      dycell(ctx, y_extent),
      xnode(ctx, x_extent + 1),
      dxnode(ctx, x_extent + 1),
      ynode(ctx, y_extent + 1),
      dynode(ctx, y_extent + 1),
      cell_volume(ctx, x_extent, y_extent),
      xface_area(ctx, x_extent + 1, y_extent),
      yface_area(ctx, x_extent, y_extent + 1),
      cell_row_pitch(x_extent), node_row_pitch(x_extent + 1), xflux_row_pitch(x_extent + 1), yface_row_stride(x_extent)

{}

subdomain_type::subdomain_type(const std::array<int, 4> &subdomainAdjacent, const int rank_id,
                       const int xLo, const int yLo, const int xHi,
                       const int yHi,
                       const int first_col, const int last_col, const int first_row, const int top,
                       const int leftOnEdge, const int rightOnEdge, const int bottomOnEdge, const int topOnEdge,
                       const int tiles_per_subdomain)
    : subdomain_adjacent(subdomainAdjacent), rank_id(rank_id),
      x_lo(xLo), y_lo(yLo), x_hi(xHi), y_hi(yHi),
      first_col(first_col), last_col(last_col), first_row(first_row), top(top),
      left_on_edge(leftOnEdge), right_on_edge(rightOnEdge), bottom_on_edge(bottomOnEdge), top_on_edge(topOnEdge) {}
sim_state::sim_state(const run_config &settings, hydro2d::context exec_ctx, subdomain_type subdomain)
    : settings(settings), context(std::move(exec_ctx)), subdomain(std::move(subdomain)), dt(settings.dt_initial), dt_prev(settings.dt_initial),
      timing_enabled(settings.timing_enabled) {}
