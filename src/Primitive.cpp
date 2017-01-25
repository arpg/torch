#include <torch/Primitive.h>
#include <torch/Context.h>
#include <torch/Exception.h>
#include <torch/Geometry.h>
#include <torch/Material.h>

namespace torch
{

Primitive::Primitive(std::shared_ptr<Context> context) :
  Node(context)
{
}

Primitive::~Primitive()
{
}

std::shared_ptr<Geometry> Primitive::GetGeometry() const
{
  return m_geometry;
}

void Primitive::SetGeometry(std::shared_ptr<Geometry> geometry)
{
  m_geometry = geometry;
  m_context->MarkDirty();
}

std::shared_ptr<Material> Primitive::GetMaterial() const
{
  return m_material;
}

void Primitive::SetMaterial(std::shared_ptr<Material> material)
{
  m_material = material;
  m_context->MarkDirty();
}

void Primitive::BuildScene(Link& link)
{
  Validate();
  Node::BuildScene(link);
  m_material->BuildScene(link);
  m_geometry->BuildScene(link);
}

void Primitive::Validate()
{
  if (!m_geometry) throw Exception("null geometry assigned to primitive");
  if (!m_material) throw Exception("null material assigned to primitive");
}

} // namespace torch