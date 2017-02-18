#include <optix.h>

RT_PROGRAM void DoNothing() { }

RT_CALLABLE_PROGRAM void SparseDoNothing(uint row, uint col, float3 value) { }