#include <torch/Light.h>

namespace torch
{

Light::Light(std::shared_ptr<Context> context) :
  Node(context)
{
}

float Light::GetLuminance() const
{
  return GetPower().GetY();
}

} // namespace torch