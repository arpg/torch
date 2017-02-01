#include <torch/MeshSampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/PtxUtil.h>
#include <torch/device/Geometry.h>

namespace torch
{

MeshSampler::MeshSampler(std::shared_ptr<Context> context) :
  GeometrySampler(context)
{
  Initialize();
}

optix::Program MeshSampler::GetProgram() const
{
  return m_program;
}

void MeshSampler::Add(const MeshData& mesh)
{
  m_meshes.push_back(mesh);
}

void MeshSampler::Clear()
{
  m_meshes.clear();
}

void MeshSampler::Update()
{
  MeshData* device;
  m_buffer->setSize(m_meshes.size());
  device = reinterpret_cast<MeshData*>(m_buffer->map());
  std::copy(m_meshes.begin(), m_meshes.end(), device);
  m_buffer->unmap();
}

void MeshSampler::Initialize()
{
  CreateProgram();
  CreateBuffer();
}

void MeshSampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("MeshSampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void MeshSampler::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["meshes"]->setBuffer(m_buffer);
  m_buffer->setFormat(RT_FORMAT_USER);
  m_buffer->setElementSize(sizeof(MeshData));
  m_buffer->setSize(0);
}

} // namespace torch