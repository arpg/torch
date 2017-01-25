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

void Geometry::UpdateTransform()
{
  m_context->MarkDirty();
}

SingleGeometry::SingleGeometry(std::shared_ptr<Context> context,
    const std::string& name) :
  Geometry(context),
  m_name(name)
{
  Initialize();
}

SingleGeometry::~SingleGeometry()
{
}

std::string SingleGeometry::GetName() const
{
  return m_name;
}

void SingleGeometry::PreBuildScene()
{
}

void SingleGeometry::BuildScene(Link& link)
{
  const Transform transform = link.GetTransform() * m_transform;
  optix::GeometryInstance instance;
  instance = m_context->CreateGeometryInstance();
  instance->setGeometry(m_geometry);
  transform.Write(instance["transform"]);
}

void SingleGeometry::PostBuildScene()
{
}

void SingleGeometry::Initialize()
{
  const std::string file = PtxUtil::GetFile(m_name);
  optix::Program intersectProgram = m_context->GetProgram(file, "Intersect");
  optix::Program boundsProgram = m_context->GetProgram(file, "GetBounds");

  m_geometry = m_context->CreateGeometry();
  m_geometry->setIntersectionProgram(intersectProgram);
  m_geometry->setBoundingBoxProgram(boundsProgram);
  m_geometry->setPrimitiveCount(1);
}

} // namespace torch