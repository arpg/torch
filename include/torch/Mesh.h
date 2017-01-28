#pragma once

#include <torch/SingleGeometry.h>

namespace torch
{

class Mesh : public SingleGeometry
{
  public:

    Mesh(std::shared_ptr<Context> context);

    void SetVertices(const std::vector<float3>& vertices);

    void SetNormals(const std::vector<float3>& normals);

    void SetFaces(const std::vector<uint3>& faces);

  protected:

    template <typename T>
    static void CopyTo(optix::Buffer buffer, const std::vector<T>& data);

  private:

    void Initialize();

    void CreateVertexBuffer();

    void CreateNormalBuffer();

    void CreateFaceBuffer();

  protected:

    optix::Buffer m_vertices;

    optix::Buffer m_normals;

    optix::Buffer m_faces;
};

} // namespace torch