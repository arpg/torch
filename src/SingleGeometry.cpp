#include <torch/SingleGeometry.h>
#include <torch/Context.h>
#include <torch/Link.h>
#include <torch/PtxUtil.h>

namespace torch
{

SingleGeometry::SingleGeometry(std::shared_ptr<Context> context,
    const std::string& name) :
  Geometry(context),
  m_name(name)
{
  Initialize();
}

std::string SingleGeometry::GetName() const
{
  return m_name;
}

void SingleGeometry::BuildScene(Link& link)
{
  Geometry::BuildScene(link);

  optix::GeometryInstance instance;
  instance = m_context->CreateGeometryInstance();
  instance->setGeometry(m_geometry);

  const Transform transform = link.GetTransform() * m_transform;
  transform.Write(instance["T_wl"]);
  transform.Inverse().Write(instance["T_lw"]);

  link.Attach(instance);
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