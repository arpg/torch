#pragma once

#include <string>
#include <torch/Node.h>
#include <torch/Spectrum.h>

namespace torch
{

class Light : public Node
{
  public:

    Light(std::shared_ptr<Context> context);

    virtual float GetLuminance() const;

    virtual Spectrum GetPower() const = 0;
};

} // namespace torch