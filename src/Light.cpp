#include <torch/Light.h>
#include <torch/Spectrum.h>

namespace torch
{

Light::Light(std::shared_ptr<Context> context) :
  Node(context)
{
}

} // namespace torch