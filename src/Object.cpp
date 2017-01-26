#include <torch/Object.h>

namespace torch
{

Object::Object(std::shared_ptr<Context> context) :
  m_context(context)
{
}

} // namespace torch