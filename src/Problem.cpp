#include <torch/Problem.h>
#include <torch/Camera.h>
#include <torch/EnvironmentLight.h>
#include <torch/Image.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/Primitive.h>
#include <torch/Scene.h>

namespace torch
{

Problem::Problem()
{
  Initialize();
}

size_t Problem::GetResidualCount() const
{
  size_t count = 0;
  unsigned int w, h;

  for (std::shared_ptr<Camera> camera : m_cameras)
  {
    camera->GetImageSize(w, h);
    count += 3 * w * h;
  }

  return count;
}

size_t Problem::GetLightParameterCount() const
{
  return 3 * m_light->GetDirectionCount();
}

size_t Problem::GetAlbedoParameterCount() const
{
  return 3 * m_material->GetAlbedoCount();
}

void Problem::ComputeLightDerivatives()
{
  ZeroAlbedoBufferSize();
  ZeroBounceImageSizes();
  SetLightBufferSize();

  Image image;

  for (std::shared_ptr<Camera> camera : m_cameras)
  {
    camera->Capture(image);
  }
}

void Problem::ComputeAlbedoDerivatives()
{
  ZeroLightBufferSize();
  SetAlbedoBufferSize();
  SetBounceImageSizes();

  Image image;

  for (std::shared_ptr<Camera> camera : m_cameras)
  {
    camera->Capture(image);
  }
}

CUdeviceptr  Problem::GetLightDerivatives()
{
  return m_lightDerivs->getDevicePointer(0);
}

CUdeviceptr Problem::GetAlbedoDerivatives()
{
  return m_albedoDerivs->getDevicePointer(0);
}

CUdeviceptr Problem::GetReferenceImages()
{
  return m_referenceImages->getDevicePointer(0);
}

CUdeviceptr Problem::GetRenderedImages()
{
  return m_renderedImages->getDevicePointer(0);
}

CUdeviceptr Problem::GetBounceImages()
{
  return m_bounceImages->getDevicePointer(0);
}

void Problem::SetLightBufferSize()
{
  optix::Context context = m_scene->GetContext();
  context["computeLightDerivs"]->setUint(1);
  const size_t w = GetLightParameterCount();
  const size_t h = GetResidualCount();
  m_lightDerivs->setSize(w, h);
}

void Problem::ZeroLightBufferSize()
{
  optix::Context context = m_scene->GetContext();
  context["computeLightDerivs"]->setUint(0);
  m_lightDerivs->setSize(1, 1);
}

void Problem::SetAlbedoBufferSize()
{
  optix::Context context = m_scene->GetContext();
  context["computeAlbedoDerivs"]->setUint(1);
  // TODO: compute sparse matrix
  // const size_t w = GetAlbedoParameterCount();
  // const size_t h = GetResidualCount();
  // m_albedoDerivs->setSize(w, h);
}

void Problem::ZeroAlbedoBufferSize()
{
  optix::Context context = m_scene->GetContext();
  context["computeAlbedoDerivs"]->setUint(0);
  m_albedoDerivs->setSize(1, 1);
}

void Problem::SetBounceImageSizes()
{
  optix::Context context = m_scene->GetContext();
  context["saveBounceImages"]->setUint(1);
  m_bounceImages->setSize(GetResidualCount());
}

void Problem::ZeroBounceImageSizes()
{
  optix::Context context = m_scene->GetContext();
  context["saveBounceImages"]->setUint(0);
  m_bounceImages->setSize(1);
}

void Problem::Initialize()
{
  CreateScene();
  CreatePrimitive();
  CreateLight();
  CreateCameras();
  CreateLightDerivBuffer();
  CreateAlbedoDerivBuffer();
  CreateReferenceImageBuffer();
  CreateRenderedImageBuffer();
  CreateBounceImageBuffer();
}

void Problem::CreateScene()
{
  m_scene = std::make_shared<Scene>();
}

void Problem::CreatePrimitive()
{
  std::shared_ptr<Primitive> primitive;
  primitive = m_scene->CreatePrimitive();

  m_mesh = m_scene->CreateMesh("shark.ply");

  m_material = std::static_pointer_cast<MatteMaterial>(
        m_scene->CreateMaterial("shark.ply"));

  primitive->SetGeometry(m_mesh);
  primitive->SetMaterial(m_material);
  m_scene->Add(primitive);
}

void Problem::CreateLight()
{
  m_light = m_scene->CreateEnvironmentLight();
  m_light->SetRowCount(21);
  m_light->SetRadiance(1E-5, 1E-5, 1E-5);
}

void Problem::CreateCameras()
{
  std::shared_ptr<Camera> camera;
  camera = m_scene->CreateCamera();
  camera->SetImageSize(640, 480);
  camera->SetFocalLength(320, 320);
  camera->SetCenterPoint(320, 240);
  m_cameras.push_back(camera);
}

void Problem::CreateLightDerivBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_lightDerivs = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_lightDerivs->setFormat(RT_FORMAT_USER);
  m_lightDerivs->setElementSize(3 * sizeof(double));
  context["lightDerivs"]->setBuffer(m_lightDerivs);
  m_lightDerivs->setSize(1, 1);
}

void Problem::CreateAlbedoDerivBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_albedoDerivs = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_albedoDerivs->setFormat(RT_FORMAT_USER);
  m_albedoDerivs->setElementSize(3 * sizeof(double));
  context["albedoDerivs"]->setBuffer(m_albedoDerivs);
  m_albedoDerivs->setSize(1, 1);
}

void Problem::CreateReferenceImageBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_referenceImages = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_referenceImages->setFormat(RT_FORMAT_FLOAT3);
  context["referenceImages"]->setBuffer(m_referenceImages);
  m_referenceImages->setSize(GetResidualCount());
}

void Problem::CreateRenderedImageBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_renderedImages = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_renderedImages->setFormat(RT_FORMAT_FLOAT3);
  context["renderedImages"]->setBuffer(m_renderedImages);
  m_renderedImages->setSize(GetResidualCount());
}

void Problem::CreateBounceImageBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_bounceImages = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_bounceImages->setFormat(RT_FORMAT_FLOAT3);
  context["bounceImages"]->setBuffer(m_bounceImages);
  m_bounceImages->setSize(0u);
}

} // namespace torch