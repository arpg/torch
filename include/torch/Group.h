#pragma once

#include <torch/Node.h>

namespace torch
{

class Group : public Node
{
  public:

    Group(std::shared_ptr<Context> context);

    ~Group();
};

} // namespace torch