#include <torch/Mesh.h>
#include <torch/Context.h>

namespace torch
{

Mesh::Mesh(std::shared_ptr<Context> context) :
  SingleGeometry(context, "Mesh")
{
  Initialize();
}

void Mesh::SetVertices(const std::vector<float3>& vertices)
{
  UpdateBounds(vertices);
  CopyTo(m_vertices, vertices);
}

void Mesh::SetNormals(const std::vector<float3>& normals)
{
  CopyTo(m_normals, normals);
}

void Mesh::SetFaces(const std::vector<uint3>& faces)
{
  m_geometry->setPrimitiveCount(faces.size());
  CopyTo(m_faces, faces);
}

BoundingBox Mesh::GetBounds(const Transform& transform)
{
  return transform * m_transform * m_bounds;
}

void Mesh::UpdateBounds(const std::vector<float3>& vertices)
{
  m_bounds = BoundingBox();

  for (const float3 vertex : vertices)
  {
    m_bounds.Union(vertex.x, vertex.y, vertex.z);
  }
}

template<typename T>
void Mesh::CopyTo(optix::Buffer buffer, const std::vector<T>& data)
{
  buffer->setSize(data.size());
  T* device = reinterpret_cast<T*>(buffer->map());
  std::copy(data.begin(), data.end(), device);
  buffer->unmap();
}

void Mesh::Initialize()
{
  CreateVertexBuffer();
  CreateNormalBuffer();
  CreateFaceBuffer();
}

void Mesh::CreateVertexBuffer()
{
  m_vertices = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["vertices"]->setBuffer(m_vertices);
  m_vertices->setFormat(RT_FORMAT_FLOAT3);
  m_vertices->setSize(0);
}

void Mesh::CreateNormalBuffer()
{
  m_normals = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["normals"]->setBuffer(m_normals);
  m_normals->setFormat(RT_FORMAT_FLOAT3);
  m_normals->setSize(0);
}

void Mesh::CreateFaceBuffer()
{
  m_faces = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["faces"]->setBuffer(m_faces);
  m_faces->setFormat(RT_FORMAT_UNSIGNED_INT3);
  m_faces->setSize(0);
}

} // namespace torch