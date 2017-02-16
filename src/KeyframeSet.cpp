#include <torch/KeyframeSet.h>
#include <torch/Context.h>
#include <torch/Keyframe.h>
#include <torch/Spectrum.h>
#include <torch/device/Camera.h>

namespace torch
{

KeyframeSet::KeyframeSet(std::shared_ptr<Context> context) :
  m_context(context),
  m_validPixelCount(0),
  m_dirty(false)
{
  Initialize();
}

KeyframeSet::~KeyframeSet()
{
}

bool KeyframeSet::Empty() const
{
  return m_keyframes.empty();
}

void KeyframeSet::Add(std::shared_ptr<Keyframe> keyframe)
{
  m_keyframes.push_back(keyframe);
  m_validPixelCount += keyframe->GetValidPixelCount();
  m_dirty = true;
}

std::shared_ptr<Keyframe> KeyframeSet::operator[](size_t index) const
{
  return m_keyframes[index];
}

size_t KeyframeSet::GetValidPixelCount() const
{
  return m_validPixelCount;
}

optix::Buffer KeyframeSet::GetCameraBuffer()
{
  UpdateBuffers();
  return m_cameras;
}

optix::Buffer KeyframeSet::GetPixelBuffer()
{
  UpdateBuffers();
  return m_pixels;
}

optix::Buffer KeyframeSet::GetReferenceBuffer()
{
  UpdateBuffers();
  return m_reference;
}

optix::Buffer KeyframeSet::GetRenderBuffer()
{
  UpdateBuffers();
  return m_render;
}

void KeyframeSet::UpdateBuffers()
{
  if (m_dirty)
  {
    UpdateCameraBuffer();
    UpdatePixelBuffer();
    UpdateReferenceBuffer();
    UpdateRenderBuffer();
    m_dirty = false;
  }
}

void KeyframeSet::UpdateCameraBuffer()
{
  m_cameras->setSize(m_keyframes.size());
  std::vector<CameraData> cameras(m_keyframes.size());

  for (size_t i = 0; i < m_keyframes.size(); ++i)
  {
    m_keyframes[i]->GetCamera(cameras[i]);
  }

  CameraData* device = reinterpret_cast<CameraData*>(m_cameras->map());
  std::copy(cameras.begin(), cameras.end(), device);
  m_cameras->unmap();
}

void KeyframeSet::UpdatePixelBuffer()
{
  m_pixels->setSize(m_validPixelCount);
  PixelSample* device = reinterpret_cast<PixelSample*>(m_pixels->map());
  std::vector<PixelSample> samples;
  std::vector<uint2> pixels;
  size_t offset = 0;

  for (size_t i = 0; i < m_keyframes.size(); ++i)
  {
    const std::shared_ptr<Keyframe> keyframe = m_keyframes[i];
    keyframe->GetValidPixels(pixels);
    samples.resize(pixels.size());

    for (size_t j = 0; j < pixels.size(); ++j)
    {
      samples[j].camera = i;
      samples[j].uv = pixels[j];
    }

    std::copy(samples.begin(), samples.end(), &device[offset]);
    offset += keyframe->GetValidPixelCount();
  }

  m_pixels->unmap();
}

void KeyframeSet::UpdateReferenceBuffer()
{
  m_reference->setSize(m_validPixelCount);
  Spectrum* device = reinterpret_cast<Spectrum*>(m_reference->map());
  std::vector<Spectrum> radiance;
  size_t offset = 0;

  for (const std::shared_ptr<Keyframe> keyframe : m_keyframes)
  {
    keyframe->GetValidPixelRadiance(radiance);
    std::copy(radiance.begin(), radiance.end(), &device[offset]);
    offset += keyframe->GetValidPixelCount();
  }

  m_reference->unmap();
}

void KeyframeSet::UpdateRenderBuffer()
{
  m_render->setSize(m_validPixelCount);
}

void KeyframeSet::Initialize()
{
  CreateCameraBuffer();
  CreatePixelBuffer();
  CreateReferenceBuffer();
  CreateRenderBuffer();
}

void KeyframeSet::CreateCameraBuffer()
{
  m_cameras = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_cameras->setFormat(RT_FORMAT_USER);
  m_cameras->setElementSize(sizeof(CameraData));
  m_cameras->setSize(0);
}

void KeyframeSet::CreatePixelBuffer()
{
  m_pixels = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_pixels->setFormat(RT_FORMAT_USER);
  m_pixels->setElementSize(sizeof(PixelSample));
  m_pixels->setSize(0);
}

void KeyframeSet::CreateReferenceBuffer()
{
  m_reference = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_reference->setFormat(RT_FORMAT_FLOAT3);
  m_reference->setSize(0);
}

void KeyframeSet::CreateRenderBuffer()
{
  m_render = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_render->setFormat(RT_FORMAT_FLOAT3);
  m_render->setSize(0);
}

} // namespace torch