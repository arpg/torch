#include <torch/Problem.h>
#include <torch/AlbedoResidualBlock.h>
#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/EnvironmentLight.h>
#include <torch/Image.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/Primitive.h>
#include <torch/PtxUtil.h>
#include <torch/ReferenceImage.h>
#include <torch/Scene.h>
#include <torch/device/Camera.h>

namespace torch
{

Problem::Problem(std::shared_ptr<Scene> scene,
    std::shared_ptr<Mesh> mesh,
    std::shared_ptr<MatteMaterial> material,
    std::shared_ptr<EnvironmentLight> light,
    const std::vector<std::shared_ptr<ReferenceImage>>& references) :
  m_scene(scene),
  m_mesh(mesh),
  m_material(material),
  m_light(light),
  m_referenceImages(references)
{
  Initialize();
}

size_t Problem::GetResidualCount() const
{
  size_t count = 0;

  for (std::shared_ptr<ReferenceImage> image : m_referenceImages)
  {
    count += 3 * image->GetValidPixelCount();
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

  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  context->Launch(m_programId, m_launchSize);
}

void Problem::ComputeAlbedoDerivatives()
{
  ZeroLightBufferSize();
  SetAlbedoBufferSize();
  SetBounceImageSizes();

  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();
  context->Launch(m_programId, m_launchSize);
}

optix::Buffer Problem::GetRenderBuffer() const
{
  return m_renderBuffer;
}

void Problem::GetRenderValues(std::vector<float3>& values)
{
  RTsize size;
  m_renderBuffer->getSize(size);
  values.resize(size);

  const float3* device = reinterpret_cast<const float3*>(m_renderBuffer->map());
  std::copy(device, device + size, values.data());
  m_renderBuffer->unmap();;
}

std::shared_ptr<SparseMatrix> Problem::GetAlbedoJacobian(size_t index) const
{
  return m_albedoBlocks[index]->GetJacobian();
}

CUdeviceptr  Problem::GetLightDerivatives()
{
  return m_lightDerivs->getDevicePointer(0);
}

CUdeviceptr Problem::GetReferenceImages()
{
  return m_referenceImageBuffer->getDevicePointer(0);
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
}

void Problem::ZeroAlbedoBufferSize()
{
  optix::Context context = m_scene->GetContext();
  context["computeAlbedoDerivs"]->setUint(0);
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
  CreateLightDerivBuffer();
  CreateReferenceImageBuffer();
  CreateRenderedImageBuffer();
  CreateBounceImageBuffer();
  CreateAlbedoBlocks();
  CreateCameraBuffer();
  CreatePixelBuffer();
  CreateRenderBuffer();
  CreateProgram();
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

void Problem::CreateReferenceImageBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_referenceImageBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_referenceImageBuffer->setFormat(RT_FORMAT_FLOAT3);
  context["referenceImages"]->setBuffer(m_referenceImageBuffer);
  m_referenceImageBuffer->setSize(GetResidualCount());
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

void Problem::CreateAlbedoBlocks()
{
  optix::Context context = m_scene->GetContext();
  m_addToAlbedoBuffer = context->createBuffer(RT_BUFFER_INPUT);
  m_addToAlbedoBuffer->setFormat(RT_FORMAT_PROGRAM_ID);
  std::vector<int> ids(m_referenceImages.size());

  for (size_t i = 0; i < m_referenceImages.size(); ++i)
  {
    std::shared_ptr<ReferenceImage> refImage = m_referenceImages[i];
    std::shared_ptr<AlbedoResidualBlock> block;
    block = std::make_shared<AlbedoResidualBlock>(m_mesh, refImage);
    m_albedoBlocks.push_back(block);
    ids[i] = block->GetAddProgram()->getId();
  }

  m_addToAlbedoBuffer->setSize(ids.size());
  int* device = reinterpret_cast<int*>(m_addToAlbedoBuffer->map());
  std::copy(ids.begin(), ids.end(), device);
  m_addToAlbedoBuffer->unmap();
  context["AddToAlbedoJacobian"]->setBuffer(m_addToAlbedoBuffer);
}

void Problem::CreateCameraBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_cameraBuffer = context->createBuffer(RT_BUFFER_INPUT);
  m_cameraBuffer->setFormat(RT_FORMAT_USER);
  m_cameraBuffer->setElementSize(sizeof(CameraData));
  std::vector<CameraData> cameras(m_referenceImages.size());

  for (size_t i = 0; i < m_referenceImages.size(); ++i)
  {
    m_referenceImages[i]->GetCamera(cameras[i]);
  }

  m_cameraBuffer->setSize(m_referenceImages.size());
  CameraData* device = reinterpret_cast<CameraData*>(m_cameraBuffer->map());
  std::copy(cameras.begin(), cameras.end(), device);
  m_cameraBuffer->unmap();
  context["cameras"]->setBuffer(m_cameraBuffer);
}

void Problem::CreatePixelBuffer()
{
  std::vector<uint2> pixels;
  std::vector<PixelSample> samples;
  PixelSample sample;

  for (size_t i = 0; i < m_referenceImages.size(); ++i)
  {
    std::shared_ptr<ReferenceImage> refImage = m_referenceImages[i];
    refImage->GetValidPixels(pixels);
    sample.camera = i;

    for (size_t j = 0; j < pixels.size(); ++j)
    {
      sample.uv = pixels[j];
      samples.push_back(sample);
    }
  }

  optix::Context context = m_scene->GetContext();
  m_pixelBuffer = context->createBuffer(RT_BUFFER_INPUT);
  m_pixelBuffer->setFormat(RT_FORMAT_USER);
  m_pixelBuffer->setElementSize(sizeof(PixelSample));
  m_pixelBuffer->setSize(samples.size());
  PixelSample* device = reinterpret_cast<PixelSample*>(m_pixelBuffer->map());
  std::copy(samples.begin(), samples.end(), device);
  m_pixelBuffer->unmap();
  context["pixelSamples"]->setBuffer(m_pixelBuffer);
  m_launchSize = samples.size();
}

void Problem::CreateRenderBuffer()
{
  optix::Context context = m_scene->GetContext();
  m_renderBuffer = context->createBuffer(RT_BUFFER_OUTPUT);
  m_renderBuffer->setFormat(RT_FORMAT_FLOAT3);
  m_renderBuffer->setSize(GetResidualCount() / 3);
}

void Problem::CreateProgram()
{
  std::shared_ptr<Context> context;
  context = m_mesh->GetContext();

  const std::string file = PtxUtil::GetFile("Problem");
  m_program = context->CreateProgram(file, "Capture");
  m_programId = context->RegisterLaunchProgram(m_program);
  m_program["render"]->setBuffer(m_renderBuffer);
}

} // namespace torch