#pragma once

#include <fstream>
#include <iostream>
#include <cstring>
#include "sh.hpp"
#include "shorder.hpp"

void cpu_initGamma();
SH<n> precise(SH<n> sh1, SH<n> sh2, SH<n> sh3, SH<n> sh4, SH<n> sh5);
SH<n> operator*(const SH<n>& a, const SH<n>& b);

