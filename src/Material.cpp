#include <torch/Material.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>
#include <torch/Ray.h>

namespace torch
{

Material::Material(std::shared_ptr<Context> context, const std::string& name) :
  Object(context),
  m_name(name)
{
  Initialize();
}

Material::~Material()
{
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
  link.Paint(m_material);
}

void Material::PostBuildScene()
{
}

void Material::Initialize()
{
  const std::string file = PtxUtil::GetFile(m_name);
  optix::Program closestHitProgram = m_context->GetProgram(file, "ClosestHit");
  optix::Program anyHitProgram = m_context->GetProgram(file, "AnyHit");

  m_material = m_context->CreateMaterial();
  m_material->setClosestHitProgram(RAY_TYPE_RADIANCE, closestHitProgram);
  m_material->setAnyHitProgram(RAY_TYPE_SHADOW, anyHitProgram);
}

} // namespace torch