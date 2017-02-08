#include <torch/Material.h>
#include <torch/Context.h>
#include <torch/Link.h>
#include <torch/PtxUtil.h>
#include <torch/device/Ray.h>

namespace torch
{

Material::Material(std::shared_ptr<Context> context, const std::string& name) :
  Object(context),
  m_name(name)
{
  Initialize();
}

std::string Material::GetName() const
{
  return m_name;
}

void Material::PreBuildScene()
{
}

void Material::BuildScene(Link& link)
{
  link.Apply(m_material);
}

void Material::PostBuildScene()
{
}

void Material::Initialize()
{
  CreateMaterial();
  CreateSharedPrograms();
  CreateUniquePrograms();
}

void Material::CreateMaterial()
{
  m_material = m_context->CreateMaterial();
}

void Material::CreateSharedPrograms()
{
  optix::Program program;
  const std::string file = PtxUtil::GetFile("Material");

  program = m_context->GetProgram(file, "ClosestHitDepth");
  m_material->setClosestHitProgram(RAY_TYPE_DEPTH, program);
}

void Material::CreateUniquePrograms()
{
  optix::Program program;
  const std::string file = PtxUtil::GetFile(m_name);

  program = m_context->GetProgram(file, "ClosestHit");
  m_material->setClosestHitProgram(RAY_TYPE_RADIANCE, program);

  program = m_context->GetProgram(file, "AnyHit");
  m_material->setAnyHitProgram(RAY_TYPE_SHADOW, program);
}

} // namespace torch