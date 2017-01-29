#pragma once

#include <torch/Node.h>

namespace torch
{

class Light : public Node
{
  public:

    Light(std::shared_ptr<Context> context);
};

} // namespace torch