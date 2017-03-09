#include <climits>
#include <gtest/gtest.h>
#include <torch/Torch.h>

namespace torch
{
namespace testing
{

inline float randf()
{
  return float(rand()) / RAND_MAX;
}

struct Vertex
{
  unsigned int index;
  float3 position;
};

TEST(Octree, General)
{
  const float treeExtent = 10.0f;
  const unsigned int maxDepth = 10;
  Octree tree(treeExtent, maxDepth);

  const float searchRadius = 0.5f;
  const float3 searchPosition = make_float3(1.5f, 0.2f, -3.1f);

  const unsigned int N = 10000;
  std::vector<Vertex> validVertices(N);
  std::vector<Vertex> invalidVertices(N);

  for (unsigned int i = 0; i < N; ++i)
  {
    Vertex vertex;
    vertex.index = i;
    vertex.position = searchPosition;

    const float radius = searchRadius * randf();
    const float theta = M_PIf * randf();
    const float phi = 2 * M_PIf * randf();
    const float r = cosf(theta);

    vertex.position.x += radius * r * cosf(phi);
    vertex.position.y += radius * r * sinf(phi);
    vertex.position.z += radius * sinf(theta);

    validVertices[i] = vertex;
    tree.AddVertex(vertex.index, vertex.position);
  }

  float maxRadius = FLT_MAX;
  maxRadius = fminf(maxRadius, treeExtent - fabs(searchPosition.x));
  maxRadius = fminf(maxRadius, treeExtent - fabs(searchPosition.y));
  maxRadius = fminf(maxRadius, treeExtent - fabs(searchPosition.z));
  const float range = maxRadius - searchRadius;

  for (unsigned int i = 0; i < N; ++i)
  {
    Vertex vertex;
    vertex.index = validVertices.size() + i;
    vertex.position = searchPosition;

    const float radius = range * randf() + searchRadius + 1E-8f;
    const float theta = M_PIf * randf();
    const float phi = 2 * M_PIf * randf();
    const float r = cosf(theta);

    vertex.position.x += radius * r * cosf(phi);
    vertex.position.y += radius * r * sinf(phi);
    vertex.position.z += radius * sinf(theta);

    invalidVertices[i] = vertex;
    tree.AddVertex(vertex.index, vertex.position);
  }

  std::vector<unsigned int> found;
  const unsigned int minIndex = validVertices.size() / 2;
  tree.GetVertices(minIndex, searchPosition, searchRadius, found);

  const unsigned int expectedCount = N - minIndex;
  ASSERT_EQ(expectedCount, found.size());

  for (size_t i = 0; i < found.size() - 1; ++i)
  {
    const Vertex& v0 = validVertices[found[i]];
    const Vertex& v1 = validVertices[found[i + 1]];
    const float d0 = length(searchPosition - v0.position);
    const float d1 = length(searchPosition - v1.position);
    ASSERT_LE(d0, d1);
  }
}

} // namespace testing

} // namespace torch
