#include <torch/ShadingRemover.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/PtxUtil.h>
#include <torch/device/Ray.h>

namespace torch
{

ShadingRemover::ShadingRemover(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<MatteMaterial> material) :
  m_mesh(mesh),
  m_material(material)
{
  Initialize();
}

ShadingRemover::~ShadingRemover()
{
}

void ShadingRemover::SetSampleCount(unsigned int count)
{
  m_sampleCount = count;
  m_program["sampleCount"]->setUint(m_sampleCount);
}

void ShadingRemover::Remove()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const size_t size = m_mesh->GetVertexCount();
  context->Launch(m_programId, size);
}

void ShadingRemover::Initialize()
{
  CreateDummyMaterial();
  CreateDummyGeometry();
  CreateProgram();
}

void ShadingRemover::CreateDummyMaterial()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const std::string file = PtxUtil::GetFile("ShadingRemover");
  optix::Program program = context->CreateProgram(file, "ClosestHit");
  m_dummyMaterial = context->CreateMaterial();
  m_dummyMaterial->setClosestHitProgram(RAY_TYPE_RADIANCE, program);
}

void ShadingRemover::CreateDummyGeometry()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const std::string file = PtxUtil::GetFile("ShadingRemover");
  optix::Program boundsProgram = context->CreateProgram(file, "GetBounds");
  optix::Program intersectProgram = context->CreateProgram(file, "Intersect");

  m_dummyGeometry = context->CreateGeometry();
  m_dummyGeometry->setBoundingBoxProgram(boundsProgram);
  m_dummyGeometry->setIntersectionProgram(intersectProgram);
  m_dummyGeometry->setPrimitiveCount(1);

  m_dummyInstance = context->CreateGeometryInstance();
  m_dummyInstance->setGeometry(m_dummyGeometry);
  m_dummyInstance->addMaterial(m_dummyMaterial);

  m_dummyAccel = context->CreateAcceleration();
  m_dummyAccel->setBuilder("NoAccel");
  m_dummyAccel->setTraverser("NoAccel");

  m_dummyGroup = context->CreateGeometryGroup();
  m_dummyGroup->setAcceleration(m_dummyAccel);
  m_dummyGroup->addChild(m_dummyInstance);
}

void ShadingRemover::CreateProgram()
{
  std::shared_ptr<Context> context = m_mesh->GetContext();
  const std::string file = PtxUtil::GetFile("ShadingRemover");
  m_program = context->CreateProgram(file, "Remove");
  m_programId = context->RegisterLaunchProgram(m_program);
  m_program["vertices"]->setBuffer(m_mesh->GetVertexBuffer());
  m_program["normals"]->setBuffer(m_mesh->GetNormalBuffer());
  m_program["albedso"]->setBuffer(m_material->GetAlbedoBuffer());
  m_program["sampleCount"]->setUint(m_sampleCount);
  m_program["dummyRoot"]->set(m_dummyGroup);
}

} // namespace torch