#include <torch/Mesh.h>
#include <torch/Context.h>
#include <torch/Point.h>
#include <torch/Normal.h>

namespace torch
{

Mesh::Mesh(std::shared_ptr<Context> context) :
  SingleGeometry(context, "Mesh")
{
  Initialize();
}

Mesh::~Mesh()
{
}

size_t Mesh::GetVertexCount() const
{
  return m_vertices.size();
}

void Mesh::GetVertices(std::vector<Point>& vertices) const
{
  vertices = m_vertices;
}

void Mesh::SetVertices(const std::vector<Point>& vertices)
{
  UpdateBounds(vertices);
  m_vertices = vertices;
  CopyTo(m_vertexBuffer, vertices);
}

void Mesh::GetNormals(std::vector<Normal>& normals) const
{
  normals = m_normals;
}

void Mesh::SetNormals(const std::vector<Normal>& normals)
{
  m_normals = normals;
  CopyTo(m_normalBuffer, normals);
}

void Mesh::GetFaces(std::vector<uint3>& faces) const
{
  faces = m_faces;
}

void Mesh::SetFaces(const std::vector<uint3>& faces)
{
  m_faces = faces;
  m_geometry->setPrimitiveCount(faces.size());
  CopyTo(m_faceBuffer, faces);
}

optix::Buffer Mesh::GetVertexBuffer() const
{
  return m_vertexBuffer;
}

optix::Buffer Mesh::GetNormalBuffer() const
{
  return m_normalBuffer;
}

optix::Buffer Mesh::GetFaceBuffer() const
{
  return m_faceBuffer;
}

BoundingBox Mesh::GetBounds(const Transform& transform)
{
  return transform * m_transform * m_bounds;
}

void Mesh::UpdateBounds(const std::vector<Point>& vertices)
{
  m_bounds = BoundingBox();

  for (const Point& vertex : vertices)
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
  m_vertexBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["vertices"]->setBuffer(m_vertexBuffer);
  m_vertexBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_vertexBuffer->setSize(0);
}

void Mesh::CreateNormalBuffer()
{
  m_normalBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["normals"]->setBuffer(m_normalBuffer);
  m_normalBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_normalBuffer->setSize(0);
}

void Mesh::CreateFaceBuffer()
{
  m_faceBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_geometry["faces"]->setBuffer(m_faceBuffer);
  m_faceBuffer->setFormat(RT_FORMAT_UNSIGNED_INT3);
  m_faceBuffer->setSize(0);
}

} // namespace torch