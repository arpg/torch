#include <torch/Link.h>
#include <torch/Context.h>

namespace torch
{

Link::Link(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

Link::~Link()
{
}

void Link::Paint(optix::Material material)
{
  m_material = material;
}

void Link::Attach(optix::Transform transform)
{
  m_group->addChild(transform);
}

void Link::Attach(optix::GeometryInstance instance)
{
  instance->setMaterialCount(0);
  instance->addMaterial(m_material);
  m_geomGroup->addChild(instance);
}

void Link::Write(optix::Variable variable) const
{
  (m_group->getChildCount() > 1) ?
      variable->set(m_group) : variable->set(m_geomGroup);
}

Link Link::Branch(const Transform& transform) const
{
  Link link(*this);
  link.m_transform = link.m_transform * transform;
  return link;
}

Transform Link::GetTransform() const
{
  return m_transform;
}

void Link::Initialize()
{
  CreateGeometryGroup();
  CreateGroup();
}

void Link::CreateGeometryGroup()
{
  m_geomGroup = m_context->CreateGeometryGroup();
  m_geomGroup->setAcceleration(CreateAcceleration());
}

void Link::CreateGroup()
{
  m_group = m_context->CreateGroup();
  m_group->setAcceleration(CreateAcceleration());
  m_group->addChild(m_geomGroup);
}

optix::Acceleration Link::CreateAcceleration()
{
  optix::Acceleration accel;
  accel = m_context->CreateAcceleration();
  accel->setBuilder("Trbvh");
  accel->setTraverser("Bvh");
  return accel;
}

} // namespace torch