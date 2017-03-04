#include <torch/Distribution1D.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

namespace torch
{

Distribution1D::Distribution1D(std::shared_ptr<Context> context,
    bool useSecondName) :
  m_useSecondName(useSecondName),
  m_context(context)
{
  Initialize();
}

optix::Program Distribution1D::GetProgram() const
{
  return m_program;
}

void Distribution1D::SetValues(const std::vector<float>& values)
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

void Distribution1D::Upload(const std::vector<float>& cdf)
{
  m_buffer->setSize(cdf.size());
  float* device = reinterpret_cast<float*>(m_buffer->map());
  std::copy(cdf.begin(), cdf.end(), device);
  m_buffer->unmap();
}

void Distribution1D::Normalize(std::vector<float>& cdf)
{
  const float integral = cdf.back();

  for (size_t i = 0; i < cdf.size(); ++i)
  {
    cdf[i] /= integral;
  }
}

void Distribution1D::Initialize()
{
  CreateProgram();
  CreateBuffer();
}

void Distribution1D::CreateProgram()
{
  const std::string name = (m_useSecondName) ? "Sample2" : "Sample";
  const std::string file = PtxUtil::GetFile("Distribution1D");
  m_program = m_context->CreateProgram(file, name);
}

void Distribution1D::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_buffer->setFormat(RT_FORMAT_FLOAT);
  m_buffer->setSize(1);
  m_program["cdf"]->setBuffer(m_buffer);
}

} // namespace torch