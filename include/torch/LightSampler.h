#pragma once

#include <memory>
#include <optixu/optixpp.h>

namespace torch
{

class Context;

class LightSampler
{
  public:

    LightSampler(std::shared_ptr<Context> context);

    ~LightSampler();

    virtual optix::Program GetProgram() const = 0;

    virtual float GetLuminance() const = 0;

    virtual void Clear() = 0;

    virtual void Update() = 0;

  protected:

    std::shared_ptr<Context> m_context;
};

} // namespace torch