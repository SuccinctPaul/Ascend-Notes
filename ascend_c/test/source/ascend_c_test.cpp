#include "lib.hpp"

auto main() -> int
{
  auto const lib = library {};

  return lib.name == "ascend_c" ? 0 : 1;
}
