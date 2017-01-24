#pragma once

#include <torch/Object.h>

namespace torch
{

class Node : public Object
{
  public:

    Node(std::shared_ptr<Context> context);

    ~Node();
};

} // namespace torch