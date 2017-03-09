#include <torch/Octree.h>
#include <torch/Exception.h>

#include <iostream>

namespace torch
{

Octree::Node::Node(float extent, unsigned int maxDepth) :
  m_origin(make_float3(0, 0, 0)),
  m_extent(extent),
  m_depth(0),
  m_maxDepth(maxDepth),
  m_indices(0),
  m_positions(0)
{
}

Octree::Node::Node(const float3& origin, float extent, unsigned int depth,
    unsigned int maxDepth) :
  m_origin(origin),
  m_extent(extent),
  m_depth(depth),
  m_maxDepth(maxDepth),
  m_indices(0),
  m_positions(0)
{
}

void Octree::Node::AddVertex(unsigned int index, const float3& position)
{
  if (m_depth == m_maxDepth)
  {
    m_indices.push_back(index);
    m_positions.push_back(position);
  }
  else
  {
    const float3 delta = position - m_origin;

    int childIndex = 0;
    childIndex |= (delta.x < 0) << 0;
    childIndex |= (delta.y < 0) << 1;
    childIndex |= (delta.z < 0) << 2;

    if (!m_children[childIndex])
    {
      float3 origin = m_origin;
      const float shift = 0.5 * m_extent;

      origin.x += (delta.x < 0) ? -shift : shift;
      origin.y += (delta.y < 0) ? -shift : shift;
      origin.z += (delta.z < 0) ? -shift : shift;

      m_children[childIndex] = std::make_unique<Node>(
            origin,
            shift,
            m_depth + 1,
            m_maxDepth);
    }

    m_children[childIndex]->AddVertex(index, position);
  }
}

void Octree::Node::GetVertices(unsigned int minIndex, const float3& position,
    float radius, std::vector<Vertex>& vertices) const
{
  if (m_depth == m_maxDepth)
  {
    Vertex vertex;

    for (size_t i = 0; i < m_indices.size(); ++i)
    {
      vertex.index = m_indices[i];

      if (vertex.index >= minIndex)
      {
        vertex.distance = length(position - m_positions[i]);

        if (vertex.distance < radius)
        {
          vertices.push_back(vertex);
        }
      }
    }
  }
  else
  {
    for (const std::unique_ptr<Node>& child : m_children)
    {
      if (child && child->Intersects(position, radius))
      {
        child->GetVertices(minIndex, position, radius, vertices);
      }
    }
  }
}

bool Octree::Node::Intersects(const float3& position, float radius) const
{
  const float voxelRadius = length(make_float3(m_extent));
  return (length(position - m_origin) - voxelRadius) < radius;
}

Octree::Octree(float extent, unsigned int maxDepth) :
  m_extent(extent),
  m_maxDepth(maxDepth),
  m_root(std::make_unique<Node>(extent, maxDepth))
{
}

Octree::~Octree()
{
}

void Octree::AddVertex(unsigned int index, const float3& position)
{
  TORCH_ASSERT(fabs(position.x) < m_extent, "position x-coord exceeds extent");
  TORCH_ASSERT(fabs(position.y) < m_extent, "position y-coord exceeds extent");
  TORCH_ASSERT(fabs(position.z) < m_extent, "position z-coord exceeds extent");

  m_root->AddVertex(index, position);
}

void Octree::GetVertices(unsigned int minIndex, const float3& position,
    float radius, std::vector<unsigned int>& indices) const
{
  std::vector<Vertex> vertices;
  m_root->GetVertices(minIndex, position, radius, vertices);
  SortVertices(vertices);
  GetIndices(vertices, indices);
}

void Octree::SortVertices(std::vector<Octree::Vertex>& vertices)
{
  std::sort(vertices.begin(), vertices.end(), Vertex());
}

void Octree::GetIndices(const std::vector<Octree::Vertex>& vertices,
    std::vector<unsigned int>& indices)
{
  indices.resize(vertices.size());

  for (size_t i = 0; i < vertices.size(); ++i)
  {
    indices[i] = vertices[i].index;
  }
}

} // namespace torch
