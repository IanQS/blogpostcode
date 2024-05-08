//
// Created by ian on 10/20/20.
//

#ifndef PALISADE_TUTORIAL_UTILS_MISC_H_
#define PALISADE_TUTORIAL_UTILS_MISC_H_

#include <iomanip>

uint nextPowerOfTwo(int n) {
  if (n <= 0) {
    return 0;
  }

  // I don't remember bit twiddling. Found this solution here:
  // https://stackoverflow.com/a/108360/3532564
  if ((n & (n - 1)) == 0) {
    // Is a power of 2 so we add 1 and get the next larger power of 2
    n += 1;
  }
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;

  return n + 1;
}
#endif //PALISADE_TUTORIAL_UTILS_MISC_H_
