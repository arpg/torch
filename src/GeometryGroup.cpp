#include <torch/GeometryGroup.h>
#include <algorithm>
#include <torch/Context.h>

namespace torch
{

GeometryGroup::GeometryGroup(std::shared_ptr<Context> context) :
  Geometry(context)
{
}

GeometryGroup::~GeometryGroup()
{
}

size_t GeometryGroup::GetChildCount() const
{
  return m_children.size();
}

std::shared_ptr<Geometry> GeometryGroup::GetChild(size_t index) const
{
  return m_children[index];
}

bool GeometryGroup::HasChild(std::shared_ptr<const Geometry> child) const
{
  auto iter = std::find(m_children.begin(), m_children.end(), child);
  return iter != m_children.end();
}

void GeometryGroup::AddChild(std::shared_ptr<Geometry> child)
{
  if (CanAddChild(child))
  {
    m_children.push_back(child);
    m_context->MarkDirty();
  }
}

void GeometryGroup::RemoveChild(std::shared_ptr<const Geometry> child)
{
  auto iter = std::find(m_children.begin(), m_children.end(), child);

  if (iter != m_children.end())
  {
    m_children.erase(iter);
    m_context->MarkDirty();
  }
}

void GeometryGroup::RemoveChildren()
{
  if (!m_children.empty())
  {
    m_children.clear();
    m_context->MarkDirty();
  }
}

void GeometryGroup::PreBuildScene()
{
}

void GeometryGroup::BuildScene(Link& link)
{
  BuildChildScene(link);
}

void GeometryGroup::PostBuildScene()
{
}

void GeometryGroup::BuildChildScene(Link& link)
{
  Link childLink = link.Branch(m_transform);

  for (std::shared_ptr<Geometry> child : m_children)
  {
    child->BuildScene(childLink);
  }
}

bool GeometryGroup::CanAddChild(std::shared_ptr<const Geometry> child) const
{
  return child && child.get() != this && !HasChild(child);
}

} // namespace torch