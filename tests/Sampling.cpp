#include <gtest/gtest.h>
#include <torch/device/Random.h>
#include <torch/device/Sampling.h>

using namespace torch;

TEST(Sampling, SampleDiskUniform)
{
  // set up bin resolution
  const unsigned int ringCount = 4;
  const unsigned int sectionCount = 8;
  const unsigned int sampleCount = 2E7;
  const unsigned int binCount = ringCount * sectionCount;
  const float ringWidth = 1.0f / ringCount;
  const float sectionRadians = 2 * M_PIf / sectionCount;

  float areas[ringCount];
  unsigned int counts[binCount];
  std::fill(counts, &counts[binCount], 0);
  unsigned int seed = init_seed<16>(0, 0);

  // pre-compute area of each ring-section
  for (unsigned int i = 0; i < ringCount; ++i)
  {
    const float innerRadius = i * ringWidth;
    const float outerRadius = (i + 1) * ringWidth;
    const float innerRadiusSq = innerRadius * innerRadius;
    const float outerRadiusSq = outerRadius * outerRadius;
    areas[i] = M_PIf * (outerRadiusSq - innerRadiusSq) / sectionCount;
  }

  // draw and bin random sample
  for (unsigned int i = 0; i < sampleCount; ++i)
  {
    float2 point;
    SampleDiskUniform(randf2(seed), point);
    const float radius = length(point);
    const float radians = M_PIf + atan2(point.y, point.x);
    const unsigned int ring = radius / ringWidth;
    const unsigned int section = radians / sectionRadians;
    const unsigned int index = ring * sectionCount + section;
    ++counts[index];
  }

  // evaluate final results
  for (unsigned int i = 0; i < binCount; ++i)
  {
    const float area = areas[i / sectionCount];
    const unsigned int expected = sampleCount * area / M_PIf;
    const unsigned int found = counts[i];
    const float ratio = float(found) / expected;
    ASSERT_NEAR(1, ratio, 5E-3);
  }
}