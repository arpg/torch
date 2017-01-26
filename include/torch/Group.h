#pragma once

#include <torch/Node.h>

namespace torch
{

class Group : public Node
{
  public:

    Group(std::shared_ptr<Context> context);

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

  protected:

    void BuildChildLink();

  protected:

    std::unique_ptr<Link> m_childLink;
};

} // namespace torch