#include <torch/Node.h>
#include <algorithm>
#include <torch/Context.h>
#include <torch/Object.h>

namespace torch
{

Node::Node(std::shared_ptr<Context> context) :
  Object(context)
{
}

Node::~Node()
{
}

size_t Node::GetChildCount() const
{
  return m_children.size();
}

std::shared_ptr<Node> Node::GetChild(size_t index) const
{
  return m_children[index];
}

bool Node::HasChild(std::shared_ptr<const Node> child) const
{
  auto iter = std::find(m_children.begin(), m_children.end(), child);
  return iter != m_children.end();
}

void Node::AddChild(std::shared_ptr<Node> child)
{
  if (CanAddChild(child))
  {
    m_children.push_back(child);
    m_context->MarkDirty();
  }
}

void Node::RemoveChild(std::shared_ptr<const Node> child)
{
  auto iter = std::find(m_children.begin(), m_children.end(), child);

  if (iter != m_children.end())
  {
    m_children.erase(iter);
    m_context->MarkDirty();
  }
}

void Node::RemoveChildren()
{
  if (!m_children.empty())
  {
    m_children.clear();
    m_context->MarkDirty();
  }
}

void Node::BuildScene(optix::Variable variable)
{
  Link link(m_context);
  BuildScene(link);
  link.Write(variable);
}

void Node::PreBuildScene()
{
}

void Node::BuildScene(Link& link)
{
  BuildChildScene(link);
}

void Node::PostBuildScene()
{
}

void Node::BuildChildScene(Link& link)
{
  Link childLink = link.Branch(m_transform);

  for (std::shared_ptr<Node> child : m_children)
  {
    child->BuildScene(childLink);
  }
}

bool Node::CanAddChild(std::shared_ptr<const Node> child) const
{
  return child && child.get() != this && !HasChild(child);
}

void Node::UpdateTransform()
{
  m_context->MarkDirty();
}

} // namespace torch