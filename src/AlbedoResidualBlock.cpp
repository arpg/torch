#include <torch/AlbedoResidualBlock.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/Mesh.h>
#include <torch/PtxUtil.h>
#include <torch/ReferenceImage.h>
#include <torch/device/Camera.h>

#include <iostream>

namespace torch
{

AlbedoResidualBlock::AlbedoResidualBlock(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<ReferenceImage> reference) :
  m_mesh(mesh),
  m_reference(reference)
{
  Initialize();
}

void AlbedoResidualBlock::Initialize()
{
  CreateAdjacencyMap();
  CreateBoundingBoxProgram();
  CreateBoundingBoxBuffer();

  std::vector<uint4> boxes;
  GetBoundingBoxes(boxes);
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
  m_bboxProgram["boundBoxes"]->setBuffer(m_bboxBuffer);
}

void AlbedoResidualBlock::GetBoundingBoxes(std::vector<uint4>& boxes)
{
  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  context->Launch(m_bboxProgramId, m_mesh->GetVertexCount());
}

} // namespace torch