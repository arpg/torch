#include <torch/Group.h>
#include <torch/Context.h>

namespace torch
{

Group::Group(std::shared_ptr<Context> context) :
  Node(context)
{
}

void Group::PreBuildScene()
{
  m_childLink.reset();
}

void Group::BuildScene(Link& link)
{
  BuildChildLink();
  optix::Transform transform;
  transform = m_context->CreateTransform();
  link.GetTransform().Write(transform);
  m_childLink->Write(transform);
  link.Attach(transform);
}

void Group::BuildChildLink()
{
  if (!m_childLink)
  {
    m_childLink = std::make_unique<Link>(m_context);
    Node::BuildChildScene(*m_childLink);
  }
}

} // namespace torch