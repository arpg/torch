#include <torch/AlbedoResidualBlock.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/Image.h>
#include <torch/Mesh.h>
#include <torch/PtxUtil.h>
#include <torch/Keyframe.h>
#include <torch/SparseMatrix.h>
#include <torch/device/Camera.h>

#include <iostream>

namespace torch
{

AlbedoResidualBlock::AlbedoResidualBlock(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<Keyframe> reference) :
  m_mesh(mesh),
  m_reference(reference)
{
  Initialize();
}

optix::Program AlbedoResidualBlock::GetAddProgram() const
{
  return m_jacobian->GetAddProgram();
}

std::shared_ptr<SparseMatrix> AlbedoResidualBlock::GetJacobian() const
{
  return m_jacobian;
}

void AlbedoResidualBlock::Initialize()
{
  CreateAdjacencyMap();
  CreateBoundingBoxProgram();
  CreateBoundingBoxBuffer();

  std::vector<uint4> boxes;
  GetBoundingBoxes(boxes);
  CreateJacobian(boxes);
}

void AlbedoResidualBlock::CreateAdjacencyMap()
{
  std::vector<Point> vertices;
  std::vector<uint3> faces;

  m_mesh->GetVertices(vertices);
  m_mesh->GetFaces(faces);

  unsigned int indexCount = 0;
  std::vector<std::vector<unsigned int>> map(vertices.size());

  for (size_t j = 0; j < faces.size(); ++j)
  {
    const unsigned int x = faces[j].x;
    const unsigned int y = faces[j].y;
    const unsigned int z = faces[j].z;

    bool canAdd1 = true;
    bool canAdd2 = true;

    for (size_t i = 0; i < map[x].size(); ++i)
    {
      if (map[x][i] == y) canAdd1 = false;
      if (map[x][i] == z) canAdd2 = false;
    }

    if (canAdd1) { map[x].push_back(y); ++indexCount; }
    if (canAdd2) { map[x].push_back(z); ++indexCount; }

    canAdd1 = true;
    canAdd2 = true;

    for (size_t i = 0; i < map[y].size(); ++i)
    {
      if (map[y][i] == x) canAdd1 = false;
      if (map[y][i] == z) canAdd2 = false;
    }

    if (canAdd1) { map[y].push_back(x); ++indexCount; }
    if (canAdd2) { map[y].push_back(z); ++indexCount; }

    canAdd1 = true;
    canAdd2 = true;

    for (size_t i = 0; i < map[z].size(); ++i)
    {
      if (map[z][i] == x) canAdd1 = false;
      if (map[z][i] == y) canAdd2 = false;
    }

    if (canAdd1) { map[z].push_back(x); ++indexCount; }
    if (canAdd2) { map[z].push_back(y); ++indexCount; }
  }

  std::vector<unsigned int> indices(indexCount);
  std::vector<unsigned int> offsets(vertices.size() + 1);
  offsets[0] = 0;

  unsigned int index = 0;

  for (size_t i = 0; i < map.size(); ++i)
  {
    for (size_t j = 0; j < map[i].size(); ++j)
    {
      indices[index++] = map[i][j];
    }

    offsets[i + 1] = index;
  }

  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();

  m_neighborOffsets = context->CreateBuffer(RT_BUFFER_INPUT);
  m_neighborOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_neighborOffsets->setSize(offsets.size());

  unsigned int* deviceOffsets =
      reinterpret_cast<unsigned int*>(m_neighborOffsets->map());

  std::copy(offsets.begin(), offsets.end(), deviceOffsets);
  m_neighborOffsets->unmap();

  m_neighborIndices = context->CreateBuffer(RT_BUFFER_INPUT);
  m_neighborIndices->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_neighborIndices->setSize(indices.size());

  unsigned int* deviceIndices =
      reinterpret_cast<unsigned int*>(m_neighborIndices->map());

  std::copy(indices.begin(), indices.end(), deviceIndices);
  m_neighborIndices->unmap();
}

void AlbedoResidualBlock::CreateBoundingBoxProgram()
{
  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();

  const std::string file = PtxUtil::GetFile("AlbedoResidualBlock");
  m_bboxProgram = context->CreateProgram(file, "GetBoundingBoxes");
  m_bboxProgramId = context->RegisterLaunchProgram(m_bboxProgram);
  m_bboxProgram["neighborOffsets"]->setBuffer(m_neighborOffsets);
  m_bboxProgram["neighborIndices"]->setBuffer(m_neighborIndices);
  m_bboxProgram["vertices"]->setBuffer(m_mesh->GetVertexBuffer());
  m_bboxProgram["normals"]->setBuffer(m_mesh->GetNormalBuffer());

  std::shared_ptr<const Camera> cam = m_reference->GetCamera();
  const Transform Twc = cam->GetTransform();
  Twc.Write(m_bboxProgram["Twc"]);
  Twc.Inverse().Write(m_bboxProgram["Tcw"]);

  CameraData camera;
  m_reference->GetCamera(camera);
  m_bboxProgram["camera"]->setUserData(sizeof(CameraData), &camera);
}

void AlbedoResidualBlock::CreateBoundingBoxBuffer()
{
  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  m_bboxBuffer = context->CreateBuffer(RT_BUFFER_OUTPUT);
  m_bboxBuffer->setFormat(RT_FORMAT_UNSIGNED_INT4);
  m_bboxBuffer->setSize(m_mesh->GetVertexCount());
  m_bboxProgram["boundingBoxes"]->setBuffer(m_bboxBuffer);
}

void AlbedoResidualBlock::GetBoundingBoxes(std::vector<uint4>& bboxes)
{
  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  context->Launch(m_bboxProgramId, m_mesh->GetVertexCount());

  bboxes.resize(m_mesh->GetVertexCount());
  uint4* device = reinterpret_cast<uint4*>(m_bboxBuffer->map());
  std::copy(device, device + bboxes.size(), bboxes.data());
  m_bboxBuffer->unmap();
}

void AlbedoResidualBlock::CreateJacobian(const std::vector<uint4>& bboxes)
{
  std::shared_ptr<const Image> mask = m_reference->GetMask();
  const unsigned int width = mask->GetWidth();
  float3* data = reinterpret_cast<float3*>(mask->GetData());

  std::vector<unsigned int> rowOffsets(bboxes.size() + 1);
  std::vector<unsigned int> colIndices;
  rowOffsets[0] = 0;

  // for each vertex
  for (size_t i = 0; i < bboxes.size(); ++i)
  {
    const uint4& bbox = bboxes[i];

    for (unsigned int y = bbox.y; y < bbox.w; ++y)
    {
      for (unsigned int x = bbox.x; x < bbox.z; ++x)
      {
        const size_t pixelIndex = y * width + x;

        if (data[pixelIndex].x == 1)
        {
          const size_t pixelIndex = m_reference->GetValidPixelIndex(x, y);
          colIndices.push_back(pixelIndex);
        }
      }
    }

    rowOffsets[i + 1] = colIndices.size();
  }

  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  m_jacobian = std::make_shared<SparseMatrix>(context);
  m_jacobian->Allocate(rowOffsets, colIndices);
}

} // namespace torch