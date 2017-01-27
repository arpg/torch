#include <torch/GeometrySampler.h>

namespace torch
{

GeometrySampler::GeometrySampler(std::shared_ptr<Context> context) :
  m_context(context)
{
}

} // namespace torch