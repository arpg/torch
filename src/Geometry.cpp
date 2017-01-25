#include <torch/Geometry.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

namespace torch
{

Geometry::Geometry(std::shared_ptr<Context> context) :
  Object(context)
{
}

Geometry::~Geometry()
{
}

void Geometry::PreBuildScene()
{
}

void Geometry::BuildScene(Link& link)
{
}

void Geometry::PostBuildScene()
{
}

void Geometry::UpdateTransform()
{
  m_context->MarkDirty();
}

} // namespace torch