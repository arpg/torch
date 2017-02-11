#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/GeometrySampler.h>
#include <torch/Group.h>
#include <torch/LightSampler.h>
#include <torch/SceneGeometrySampler.h>
#include <torch/SceneLightSampler.h>
#include <torch/device/Camera.h>
#include <torch/device/Ray.h>

#include <string.h>
#include <iostream>
#include <torch/PtxUtil.h>

namespace torch
{

Context::Context() :
  m_dirty(true)
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
  FinishLaunch();
}

void Context::Launch(unsigned int id, RTsize w, RTsize h)
{
  PrepareLaunch();
  m_context->launch(id, w, h);
  FinishLaunch();
}

void Context::Launch(unsigned int id, RTsize w, RTsize h, RTsize d)
{
  PrepareLaunch();
  m_context->launch(id, w, h, d);
  FinishLaunch();
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

#ifdef DEBUG_BUILD
  m_context->setExceptionProgram(id, m_errorProgram);
#endif

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

std::shared_ptr<SceneLightSampler> Context::GetLightSampler() const
{
  return m_lightSampler;
}

std::shared_ptr<SceneGeometrySampler> Context::GetGeometrySampler() const
{
  return m_geometrySampler;
}

BoundingBox Context::GetSceneBounds() const
{
  return m_bounds;
}

float Context::GetSceneRadius() const
{
  return m_bounds.GetRadius();
}

optix::Context Context::GetContext()
{
  return m_context;
}

void Context::PrepareLaunch()
{
  if (m_dirty)
  {
    m_lightSampler->Clear();
    DropOrphans();
    PreBuildScene();
    m_bounds = m_sceneRoot->GetBounds(Transform());
    BuildScene();
    PostBuildScene();
    m_lightSampler->Update();
    m_dirty = false;

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

void Context::FinishLaunch()
{
#ifdef DEBUG_BUILD
  unsigned char host[128];
  unsigned char* device = reinterpret_cast<unsigned char*>(m_errorBuffer->map());
  std::copy(device, device + 128, host);
  m_errorBuffer->unmap();

  if (strcmp(reinterpret_cast<const char*>(host), "device stack overflow") == 0)
  {
    std::cerr << "Message: " << host << std::endl;
  }
#endif
}

void Context::Initialize()
{
  CreateContext();
  CreateSceneRoot();
  CreateLightSampler();
  CreateGeometrySampler();

  optix::Buffer addToAlbedoBuffer;
  addToAlbedoBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
  addToAlbedoBuffer->setFormat(RT_FORMAT_PROGRAM_ID);
  addToAlbedoBuffer->setSize(0);
  m_context["AddToAlbedoJacobian"]->setBuffer(addToAlbedoBuffer);

  optix::Buffer pixelSamples;
  pixelSamples = m_context->createBuffer(RT_BUFFER_INPUT);
  pixelSamples->setFormat(RT_FORMAT_USER);
  pixelSamples->setElementSize(sizeof(PixelSample));
  pixelSamples->setSize(0);
  m_context["pixelSamples"]->setBuffer(pixelSamples);

  optix::Buffer cameras;
  cameras = m_context->createBuffer(RT_BUFFER_INPUT);
  cameras->setFormat(RT_FORMAT_USER);
  cameras->setElementSize(sizeof(CameraData));
  cameras->setSize(0);
  m_context["cameras"]->setBuffer(cameras);

  optix::Buffer lightDerivatives;
  lightDerivatives = m_context->createBuffer(RT_BUFFER_OUTPUT);
  lightDerivatives->setFormat(RT_FORMAT_FLOAT3);
  lightDerivatives->setSize(1, 1);
  m_context["lightDerivatives"]->setBuffer(lightDerivatives);

  m_context["computeLightDerivs"]->setUint(0);
  m_context["computeAlbedoDerivs"]->setUint(0);
}

void Context::CreateContext()
{
  m_context = optix::Context::create();
  m_context->setRayTypeCount(RAY_TYPE_COUNT);
  m_context["sceneEpsilon"]->setFloat(1E-4);
  m_context->setStackSize(2048);

#ifdef DEBUG_BUILD
  m_context->setPrintEnabled(true);
  // m_context->setPrintBufferSize(512);
  m_context->setPrintBufferSize(8192);
  m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true);

  const std::string file = PtxUtil::GetFile("Exception");
  m_errorProgram = m_context->createProgramFromPTXFile(file, "HandleError");
  m_errorBuffer = m_context->createBuffer(RT_BUFFER_OUTPUT);
  m_errorBuffer->setFormat(RT_FORMAT_UNSIGNED_BYTE);
  m_errorBuffer->setSize(128);
  m_errorProgram["buffer"]->setBuffer(m_errorBuffer);
#endif
}

void Context::CreateSceneRoot()
{
  m_sceneRoot = std::make_shared<Group>(shared_from_this());
}

void Context::CreateLightSampler()
{
  m_lightSampler = std::make_shared<SceneLightSampler>(shared_from_this());
  m_context["SampleLights"]->set(m_lightSampler->GetProgram());
}

void Context::CreateGeometrySampler()
{
  std::shared_ptr<Context> sharedThis = shared_from_this();
  m_geometrySampler = std::make_shared<SceneGeometrySampler>(sharedThis);
  m_context["SampleGeometry"]->set(m_geometrySampler->GetProgram());
}

} // namespace torch