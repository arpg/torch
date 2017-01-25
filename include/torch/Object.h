#pragma once

#include <memory>
#include <torch/Link.h>

namespace torch
{

class Context;

class Object
{
  public:

    Object(std::shared_ptr<Context> context);

    ~Object();

    virtual void PreBuildScene() = 0;

    virtual void BuildScene(Link& link) = 0;

    virtual void PostBuildScene() = 0;

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch