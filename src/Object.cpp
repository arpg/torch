#include <torch/Object.h>

namespace torch
{

Object::Object(std::shared_ptr<Context> context) :
  m_context(context)
{
}

std::shared_ptr<Context> Object::GetContext() const
{
  return m_context;
}

} // namespace torch