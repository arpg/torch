#include <torch/Group.h>

namespace torch
{

Group::Group(std::shared_ptr<Context> context) :
  Node(context)
{
}

Group::~Group()
{
}

} // namespace torch