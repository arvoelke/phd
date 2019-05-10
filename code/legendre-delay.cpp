/* Copyright 2019 Aaron R. Voelker */

#include <iostream>
#include <iomanip>
#include <limits>

typedef long double float_t;
const float_t EPS = 1e-12;
const char FILL[] = "\e[K";

void print_x(const float_t x[], const int size) {
  for (int i=0; i < size; ++i) {
    std::cout << x[i];
    if (i != size-1) std::cout << ",";
  }
  std::cout << std::endl;
}

void simulate(const int steps,
              const float_t theta, const int q,
              const float_t dt, const int sample_every) {
  const float_t input = 1;  // TODO(arvoelke): generalize
  float_t x[q] = {0};
  float_t dx[q] = {0};
  float_t l, u;
  for (int k=0; k < steps; ++k) {
    if (k % sample_every == 0) {  // IO
      print_x(x, q);
      std::cerr << "\r" << std::setprecision(2) << std::fixed <<
        100. * k / steps << "%" << FILL << std::flush;
    }
    for (int i=0; i < q; ++i) {  // diagonals
      dx[i] = -x[i];
    }
    l = -input;
    dx[0] += input;
    for (int i=0; i < q-1; ++i) {  // lower triangle
      l = (-l) - x[i];
      dx[i+1] -= l;
    }
    u = 0;
    for (int i=q-1; i > 0; --i) {  // upper triangle
      u = (+u) + x[i];
      dx[i-1] -= u;
    }
    for (int i=0; i < q; ++i) {  // integrate
      x[i] += dt*dx[i]*(2.*i + 1.)/theta;
    }
  }
  std::cerr << "\r100%" << FILL << std::endl;
}

int main(int argc, const char** argv) {
  const int q = 10240;
  const float_t dt = 1e-5;  // must be small for Euler's method
  const int steps = q/dt + EPS;
  const int samples = 1e3;

  std::cout.precision(std::numeric_limits< float_t >::max_digits10);
  simulate(steps, q/2, q, dt, steps / samples);
  return 0;
}

