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
  m_material = m_context->CreateMaterial();
  const std::string file = PtxUtil::GetFile(m_name);

  optix::Program closestHitProgram = m_context->GetProgram(file, "ClosestHit");
  m_material->setClosestHitProgram(RAY_TYPE_RADIANCE, closestHitProgram);

  optix::Program anyHitProgram = m_context->GetProgram(file, "AnyHit");
  m_material->setAnyHitProgram(RAY_TYPE_SHADOW, anyHitProgram);
}

} // namespace torch