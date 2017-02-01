#include <torch/GeometryGroupSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/PtxUtil.h>
#include <torch/device/Geometry.h>

namespace torch
{

GeometryGroupSampler::GeometryGroupSampler(std::shared_ptr<Context> context) :
  GeometrySampler(context)
{
  Initialize();
}

optix::Program GeometryGroupSampler::GetProgram() const
{
  return m_program;
}

void GeometryGroupSampler::Add(const GeometryGroupData& group)
{
  m_groups.push_back(group);
}

void GeometryGroupSampler::Clear()
{
  m_groups.clear();
}

void GeometryGroupSampler::Update()
{
  GeometryGroupData* device;
  m_buffer->setSize(m_groups.size());
  device = reinterpret_cast<GeometryGroupData*>(m_buffer->map());
  std::copy(m_groups.begin(), m_groups.end(), device);
  m_buffer->unmap();
}

void GeometryGroupSampler::Initialize()
{
  CreateProgram();
  CreateBuffer();
}

void GeometryGroupSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("GeometryGroupSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void GeometryGroupSampler::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["groups"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(GeometryGroupData));
  m_buffer->setSize(0);
}

} // namespace torch