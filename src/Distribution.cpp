#include <torch/Distribution.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

namespace torch
{

Distribution::Distribution(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

optix::Program Distribution::GetProgram() const
{
  return m_program;
}

void Distribution::SetValues(const std::vector<float>& values)
{
  std::vector<float> cdf(values.size() + 1);
  cdf[0] = 0;

  for (size_t i = 0; i < values.size(); ++i)
  {
    cdf[i + 1] = values[i] + cdf[i];
  }

  Normalize(cdf);
  Upload(cdf);
}

void Distribution::Upload(const std::vector<float>& cdf)
{
  m_buffer->setSize(cdf.size());
  float* device = reinterpret_cast<float*>(m_buffer->map());
  std::copy(cdf.begin(), cdf.end(), device);
  m_buffer->unmap();
}

void Distribution::Normalize(std::vector<float>& cdf)
{
  const float integral = cdf.back();

  for (size_t i = 0; i < cdf.size(); ++i)
  {
    cdf[i] /= integral;
  }
}

void Distribution::Initialize()
{
  CreateBuffer();
  CreateProgram();
}

void Distribution::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_buffer->setFormat(RT_FORMAT_FLOAT);
  m_buffer->setSize(1);
}

void Distribution::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("Distribution");
  m_program = m_context->CreateProgram(file, "Sample");
  m_program["cdf"]->setBuffer(m_buffer);
}

} // namespace torch