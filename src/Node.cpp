#include <torch/Node.h>
#include <torch/Object.h>

namespace torch
{

Node::Node(std::shared_ptr<Context> context) :
  Object(context)
{
}

Node::~Node()
{
}

} // namespace torch