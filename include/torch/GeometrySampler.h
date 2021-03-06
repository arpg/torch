#pragma once

#include <torch/Core.h>

namespace torch
{

class GeometrySampler
{
  public:

    GeometrySampler(std::shared_ptr<Context> context);

    virtual optix::Program GetProgram() const = 0;

    virtual void Clear() = 0;

    virtual void Update() = 0;

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch