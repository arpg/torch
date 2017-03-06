#pragma once

#include <torch/Core.h>

namespace torch
{

class Octree
{
  private:

    struct Vertex
    {
      unsigned int index;

      float distance;

      inline bool operator()(const Vertex& a, const Vertex& b)
      {
        return a.distance < b.distance;
      }
    };

    class Node
    {
      public:

        Node(float extent, unsigned int maxDepth);

        Node(const float3& origin, float extent, unsigned int depth,
            unsigned int maxDepth);

        void AddVertex(unsigned int index, const float3& position);

        void GetVertices(unsigned int minIndex, const float3& position,
            float radius, std::vector<Vertex>& vertices) const;

        bool Intersects(const float3& position, float radius) const;

      protected:

        const float3 m_origin;

        const float m_extent;

        const unsigned int m_depth;

        const unsigned int m_maxDepth;

        std::unique_ptr<Node> m_children[8];

        std::vector<unsigned int> m_indices;

        std::vector<float3> m_positions;
    };

  public:

    Octree(float extent, unsigned int maxDepth);

    virtual ~Octree();

    void AddVertex(unsigned int index, const float3& position);

    void GetVertices(unsigned int minIndex, const float3& position,
        float radius, std::vector<unsigned int>& indices) const;

  protected:

    static void SortVertices(std::vector<Vertex>& vertices);

    static void GetIndices(const std::vector<Vertex>& vertices,
        std::vector<unsigned int>& indices);

  protected:

    const float m_extent;

    const unsigned int m_maxDepth;

    std::unique_ptr<Node> m_root;
};

} // namespace torch