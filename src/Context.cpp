#include <torch/Context.h>
#include <torch/Ray.h>

namespace torch
{

Context::Context() :
  m_dirty(true)
{
}

Context::~Context()
{
}

void Context::MarkDirty()
{
  m_dirty = true;
}

void Context::Launch(unsigned int id, RTsize w)
{
  PrepareLaunch();
  m_context->launch(id, w);
}

void Context::Launch(unsigned int id, RTsize w, RTsize h)
{
  PrepareLaunch();
  m_context->launch(id, w, h);
}

void Context::Launch(unsigned int id, RTsize w, RTsize h, RTsize d)
{
  PrepareLaunch();
  m_context->launch(id, w, h, d);
}

void Context::Launch(unsigned int id, const uint2& size)
{
  Launch(id, size.x, size.y);
}

void Context::Launch(unsigned int id, const uint3& size)
{
  Launch(id, size.x, size.y, size.z);
}

std::shared_ptr<Context> Context::Create()
{
  std::shared_ptr<Context> context;
  context = std::shared_ptr<Context>(new Context);
  context->Initialize();
  return context;
}

optix::Buffer Context::CreateBuffer(unsigned int type)
{
  return m_context->createBuffer(type);
}

optix::Program Context::CreateProgram(const std::string& file,
    const std::string& name)
{
  return m_context->createProgramFromPTXFile(file, name);
}

unsigned int Context::RegisterLaunchProgram(optix::Program program)
{
  const unsigned int id = m_context->getEntryPointCount();
  m_context->setEntryPointCount(id + 1);
  m_context->setRayGenerationProgram(id, program);
  return id;
}

void Context::PrepareLaunch()
{
  if (m_dirty)
  {
    m_dirty = true;

#ifdef DEBUG_BUILD
    m_context->validate();
    m_context->compile();
#endif
  }
}

void Context::Initialize()
{
  CreateContext();
}

void Context::CreateContext()
{
  m_context = optix::Context::create();
  m_context->setRayTypeCount(RAY_TYPE_COUNT);

#ifdef DEBUG_BUILD
  m_context->setPrintEnabled(true);
  m_context->setPrintBufferSize(512);
  m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif
}

} // namespace torch