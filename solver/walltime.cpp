

#include <cstdlib>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/times.h>

double wall_seconds() {
  timeval t{};
  gettimeofday(&t, (struct timezone *)nullptr);
  return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1.0e-6;
}
