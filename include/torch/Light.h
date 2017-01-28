#pragma once

#include <torch/Node.h>

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