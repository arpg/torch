#pragma once

#include <torch/SingleGeometry.h>
#include <torch/BoundingBox.h>

namespace torch
{

class Mesh : public SingleGeometry
{
  public:

    Mesh(std::shared_ptr<Context> context);

    ~Mesh();

    size_t GetVertexCount() const;

    void GetVertices(std::vector<Point>& vertices) const;

    void SetVertices(const std::vector<Point>& vertices);

    void GetNormals(std::vector<Normal>& normals) const;

    void SetNormals(const std::vector<Normal>& normals);

    void GetFaces(std::vector<uint3>& faces) const;

    void SetFaces(const std::vector<uint3>& faces);

    optix::Buffer GetVertexBuffer() const;

    optix::Buffer GetNormalBuffer() const;

    optix::Buffer GetFaceBuffer() const;

    void GetVertexAdjacencyMap(std::vector<uint>& map,
        std::vector<uint>& offsets, bool includeSelf) const;

    void GetVertexAdjacencyMap(std::vector<std::vector<uint>>& map,
        bool includeSelf) const;

    BoundingBox GetBounds(const Transform& transform) override;

  protected:

    static void AddAdjacencies(std::vector<std::vector<uint>>& map,
        const uint3& face, uint index, bool includeSelf);

    static void AddAdjacencies(std::vector<uint>& map, uint parent, uint child);

    void UpdateBounds(const std::vector<Point>& vertices);

    template <typename T>
    static void CopyTo(optix::Buffer buffer, const std::vector<T>& data);

  private:

    void Initialize();

    void CreateVertexBuffer();

    void CreateNormalBuffer();

    void CreateFaceBuffer();

  protected:

    std::vector<Point> m_vertices;

    std::vector<Normal> m_normals;

    std::vector<uint3> m_faces;

    optix::Buffer m_vertexBuffer;

    optix::Buffer m_normalBuffer;

    optix::Buffer m_faceBuffer;

    BoundingBox m_bounds;
};

} // namespace torch