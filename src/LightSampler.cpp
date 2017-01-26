#include <torch/LightSampler.h>

namespace torch
{

LightSampler::LightSampler(std::shared_ptr<Context> context) :
  m_context(context)
{
}

LightSampler::~LightSampler()
{
}

} // namespace torch