#include <torch/SphereSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/PtxUtil.h>
#include <torch/device/Geometry.h>

namespace torch
{

SphereSampler::SphereSampler(std::shared_ptr<Context> context) :
  GeometrySampler(context)
{
  Initialize();
}

optix::Program SphereSampler::GetProgram() const
{
  return m_program;
}

void SphereSampler::Add(const SphereData& sphere)
{
  m_spheres.push_back(sphere);
}

void SphereSampler::Clear()
{
  m_spheres.clear();
}

void SphereSampler::Update()
{
  SphereData* device;
  m_buffer->setSize(m_spheres.size());
  device = reinterpret_cast<SphereData*>(m_buffer->map());
  std::copy(m_spheres.begin(), m_spheres.end(), device);
  m_buffer->unmap();
}

void SphereSampler::Initialize()
{
  CreateProgram();
  CreateBuffer();
}

void SphereSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("SphereSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void SphereSampler::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["spheres"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(SphereData));
  m_buffer->setSize(0);
}

} // namespace torch