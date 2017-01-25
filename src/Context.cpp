#include <torch/Context.h>
#include <torch/Group.h>
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

optix::Acceleration Context::CreateAcceleration()
{
  return m_context->createAcceleration("NoAccel", "NoAccel");
}

optix::Buffer Context::CreateBuffer(unsigned int type)
{
  return m_context->createBuffer(type);
}

optix::Group Context::CreateGroup()
{
  return m_context->createGroup();
}

optix::Geometry Context::CreateGeometry()
{
  return m_context->createGeometry();
}

optix::GeometryGroup Context::CreateGeometryGroup()
{
  return m_context->createGeometryGroup();
}

optix::GeometryInstance Context::CreateGeometryInstance()
{
  return m_context->createGeometryInstance();
}

optix::Material Context::CreateMaterial()
{
  return m_context->createMaterial();
}

optix::Transform Context::CreateTransform()
{
  return m_context->createTransform();
}

optix::Variable Context::GetVariable(const std::__cxx11::string& name)
{
  return m_context[name];
}

optix::Program Context::GetProgram(const std::string& file,
    const std::string& name)
{
  const std::string key = file + "::" + name;
  optix::Program program = m_programs[key];

  if (!program)
  {
    program = CreateProgram(file, name);
    m_programs[key] = program;
  }

  return program;
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

void Context::RegisterObject(std::shared_ptr<Object> object)
{
  m_objects.push_back(object);
}

std::shared_ptr<Node> Context::GetSceneRoot() const
{
  return m_sceneRoot;
}

void Context::PrepareLaunch()
{
  if (m_dirty)
  {
    DropOrphans();
    PreBuildScene();
    BuildScene();
    PostBuildScene();
    m_dirty = true;

#ifdef DEBUG_BUILD
    m_context->validate();
    m_context->compile();
#endif
  }
}

void Context::DropOrphans()
{
  size_t index = 0;

  for (const std::shared_ptr<Object>& object : m_objects)
  {
    if (object.use_count() > 1) m_objects[index++] = object;
  }

  m_objects.resize(index);
}

void Context::PreBuildScene()
{
  for (std::shared_ptr<Object> object : m_objects)
  {
    object->PreBuildScene();
  }
}

void Context::BuildScene()
{
  optix::Variable variable;
  variable = m_context["sceneRoot"];
  m_sceneRoot->BuildScene(variable);
}

void Context::PostBuildScene()
{
  for (std::shared_ptr<Object> object : m_objects)
  {
    object->PostBuildScene();
  }
}

void Context::Initialize()
{
  CreateContext();
  CreateSceneRoot();
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

void Context::CreateSceneRoot()
{
  m_sceneRoot = std::make_shared<Group>(shared_from_this());
}

} // namespace torch