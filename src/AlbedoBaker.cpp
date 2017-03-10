#include <torch/AlbedoBaker.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/PtxUtil.h>
#include <torch/Scene.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Ray.h>

namespace torch
{

AlbedoBaker::AlbedoBaker(std::shared_ptr<Scene> scene) :
  m_scene(scene),
  m_sampleCount(1)
{
  Initialize();
}

unsigned int AlbedoBaker::GetSampleCount() const
{
  return m_sampleCount;
}

void AlbedoBaker::SetSampleCount(unsigned int count)
{
  m_sampleCount = count;
  m_bakeProgram["sampleCount"]->setUint(m_sampleCount);
}

void AlbedoBaker::Bake(std::shared_ptr<MatteMaterial> material,
    std::shared_ptr<Mesh> mesh)
{
  RTsize size;
  material->GetAlbedoBuffer()->getSize(size);
  m_scratchBuffer->setSize(size);

  m_bakeProgram["vertices"]->setBuffer(mesh->GetVertexBuffer());
  m_bakeProgram["normals"]->setBuffer(mesh->GetNormalBuffer());
  m_bakeProgram["albedos"]->setBuffer(material->GetAlbedoBuffer());
  m_copyProgram["albedos"]->setBuffer(material->GetAlbedoBuffer());
  m_material["vertices"]->setBuffer(mesh->GetVertexBuffer());
  m_material["normals"]->setBuffer(mesh->GetNormalBuffer());
  m_material["albedos"]->setBuffer(material->GetAlbedoBuffer());

  std::shared_ptr<Context> context = m_scene->GetContext();
  context->Launch(m_bakeProgramId, size);
  context->Launch(m_copyProgramId, size);
  material->LoadAlbedos();
}

void AlbedoBaker::Initialize()
{
  CreateMaterial();
  CreateGeometry();
  CreateScratchBuffer();
  CreateBakeProgram();
  CreateCopyProgram();
}

void AlbedoBaker::CreateMaterial()
{
  const std::string file = PtxUtil::GetFile("AlbedoBaker");
  std::shared_ptr<Context> context = m_scene->GetContext();
  m_hitProgram = context->CreateProgram(file, "ClosestHit");
  m_material = context->CreateMaterial();
  m_material->setClosestHitProgram(RAY_TYPE_RADIANCE, m_hitProgram);
}

void AlbedoBaker::CreateGeometry()
{
  const std::string file = PtxUtil::GetFile("AlbedoBaker");
  std::shared_ptr<Context> context = m_scene->GetContext();
  optix::Program boundsProgram = context->CreateProgram(file, "GetBounds");
  optix::Program intersectProgram = context->CreateProgram(file, "Intersect");

  m_geometry = context->CreateGeometry();
  m_geometry->setBoundingBoxProgram(boundsProgram);
  m_geometry->setIntersectionProgram(intersectProgram);
  m_geometry->setPrimitiveCount(1);

  m_instance = context->CreateGeometryInstance();
  m_instance->setGeometry(m_geometry);
  m_instance->addMaterial(m_material);

  m_accel = context->CreateAcceleration();
  m_group = context->CreateGeometryGroup();
  m_group->setAcceleration(m_accel);
  m_group->addChild(m_instance);
}

void AlbedoBaker::CreateScratchBuffer()
{
  std::shared_ptr<Context> context = m_scene->GetContext();
  m_scratchBuffer = context->CreateBuffer(RT_BUFFER_OUTPUT);
  m_scratchBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_scratchBuffer->setSize(0);
}

void AlbedoBaker::CreateBakeProgram()
{
  const std::string file = PtxUtil::GetFile("AlbedoBaker");
  std::shared_ptr<Context> context = m_scene->GetContext();
  m_bakeProgram = context->CreateProgram(file, "Bake");
  m_bakeProgramId = context->RegisterLaunchProgram(m_bakeProgram);
  m_bakeProgram["scratch"]->setBuffer(m_scratchBuffer);
  m_bakeProgram["bakeRoot"]->set(m_group);
}

void AlbedoBaker::CreateCopyProgram()
{
  const std::string file = PtxUtil::GetFile("AlbedoBaker");
  std::shared_ptr<Context> context = m_scene->GetContext();
  m_copyProgram = context->CreateProgram(file, "Copy");
  m_copyProgramId = context->RegisterLaunchProgram(m_copyProgram);
  m_copyProgram["scratch"]->setBuffer(m_scratchBuffer);
}

} // namespace torch