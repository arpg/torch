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

void Mesh::GetVertexAdjacencyMap(std::vector<uint>& map,
    std::vector<uint>& offsets, bool includeSelf) const
{
  std::vector<std::vector<uint>> map2D;
  GetVertexAdjacencyMap(map2D, includeSelf);
  offsets.resize(map2D.size() + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < map2D.size(); ++i)
  {
    const std::vector<uint>& row = map2D[i];
    offsets[i + 1] = row.size() + offsets[i];
  }

  map.resize(offsets.back());

  for (size_t i = 0; i < map2D.size(); ++i)
  {
    const std::vector<uint>& row = map2D[i];
    std::copy(row.begin(), row.end(), &map[offsets[i]]);
  }
}

void Mesh::GetVertexAdjacencyMap(std::vector<std::vector<uint>>& map,
    bool includeSelf) const
{
  map.resize(m_vertices.size());

  for (const uint3& face : m_faces)
  {
    AddAdjacencies(map, face, 0, includeSelf);
    AddAdjacencies(map, face, 1, includeSelf);
    AddAdjacencies(map, face, 2, includeSelf);
  }
}

BoundingBox Mesh::GetBounds(const Transform& transform)
{
  return transform * m_transform * m_bounds;
}

void Mesh::AddAdjacencies(std::vector<std::vector<uint>>& map,
    const uint3& face, uint index, bool includeSelf)
{
  const uint child1 = (index + 1) % 3;
  const uint child2 = (index + 2) % 3;
  const uint* array = reinterpret_cast<const uint*>(&face);
  AddAdjacencies(map[array[index]], array[index], array[child1]);
  AddAdjacencies(map[array[index]], array[index], array[child2]);

  if (includeSelf)
  {
    AddAdjacencies(map[array[index]], array[index], array[index]);
  }
}

void Mesh::AddAdjacencies(std::vector<uint>& map, uint parent, uint child)
{
  auto iter = std::find(map.begin(), map.end(), child);
  if (iter == map.end()) map.push_back(child);
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