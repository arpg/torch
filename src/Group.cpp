#include <torch/Group.h>
#include <torch/Context.h>

namespace torch
{

Group::Group(std::shared_ptr<Context> context) :
  Node(context)
{
}

} // namespace torch