#include <torch/Keyframe.h>
#include <torch/Camera.h>
#include <torch/Image.h>
#include <torch/Spectrum.h>

namespace torch
{

Keyframe::Keyframe(std::shared_ptr<Camera> camera,
    std::shared_ptr<Image> image) :
  m_camera(camera),
  m_image(image)
{
  Initialize();
}

void Keyframe::GetCamera(CameraData& camera) const
{
  m_camera->GetData(camera);
}

std::shared_ptr<const Camera> Keyframe::GetCamera() const
{
  return m_camera;
}

std::shared_ptr<const Image> Keyframe::GetImage() const
{
  return m_image;
}

std::shared_ptr<const Image> Keyframe::GetMask() const
{
  return m_mask;
}

size_t Keyframe::GetValidPixelCount() const
{
  return m_validPixelCount;
}

void Keyframe::GetValidPixels(std::vector<uint2>& pixels) const
{
  float3* data = reinterpret_cast<float3*>(m_mask->GetData());
  pixels.clear();

  for (unsigned int y = 0; y < m_mask->GetHeight(); ++y)
  {
    for (unsigned int x = 0; x < m_mask->GetWidth(); ++x)
    {
      const size_t index = y * m_mask->GetWidth() + x;
      if (data[index].x == 1) pixels.push_back(make_uint2(x, y));
    }
  }
}

size_t Keyframe::GetValidPixelIndex(unsigned int x, unsigned int y) const
{
  const size_t index = y * m_mask->GetWidth() + x;
  const size_t validIndex = m_validPixelMap[index];
  const size_t totalPixels = m_mask->GetHeight() * m_mask->GetWidth();

  if (validIndex == totalPixels)
  {
    throw "pixel is not valid";
  }

  return validIndex;
}

void Keyframe::GetValidPixelRadiance(std::vector<Spectrum>& radiance)
{
  size_t i = 0;
  radiance.resize(m_validPixelCount);
  const size_t totalPixels = m_mask->GetHeight() * m_mask->GetWidth();
  Spectrum* data = reinterpret_cast<Spectrum*>(m_image->GetData());

  for (unsigned int y = 0; y < m_mask->GetHeight(); ++y)
  {
    for (unsigned int x = 0; x < m_mask->GetWidth(); ++x)
    {
      const size_t index = y * m_mask->GetWidth() + x;

      if (m_validPixelMap[index] != totalPixels)
      {
        radiance[i++] = data[index];
      }
    }
  }
}

void Keyframe::Initialize()
{
  CreateImageMask();
  CreateValidPixelMap();
}

void Keyframe::CreateImageMask()
{
  m_mask = std::make_shared<Image>();
  m_camera->CaptureMask(*m_mask);
  m_mask->Save("temp.png");
}

void Keyframe::CreateValidPixelMap()
{
  const size_t totalPixels = m_mask->GetHeight() * m_mask->GetWidth();
  m_validPixelMap.resize(totalPixels);

  float3* data = reinterpret_cast<float3*>(m_mask->GetData());
  m_validPixelCount = 0;

  for (unsigned int y = 0; y < m_mask->GetHeight(); ++y)
  {
    for (unsigned int x = 0; x < m_mask->GetWidth(); ++x)
    {
      const size_t index = y * m_mask->GetWidth() + x;

      if (data[index].x == 1)
      {
        m_validPixelMap[index] = m_validPixelCount;
        ++m_validPixelCount;
      }
      else
      {
        m_validPixelMap[index] = totalPixels;
      }
    }
  }
}

} // namespace torch