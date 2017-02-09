#pragma once

#include <torch/Core.h>

namespace torch
{

class Object
{
  public:

    Object(std::shared_ptr<Context> context);

    std::shared_ptr<Context> GetContext() const;

    virtual void PreBuildScene() = 0;

    virtual void BuildScene(Link& link) = 0;

    virtual void PostBuildScene() = 0;

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch