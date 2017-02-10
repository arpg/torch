#include <torch/ReferenceImage.h>
#include <torch/Camera.h>
#include <torch/Image.h>

namespace torch
{

ReferenceImage::ReferenceImage(std::shared_ptr<Camera> camera,
    std::shared_ptr<Image> image) :
  m_camera(camera),
  m_image(image)
{
  Initialize();
}

void ReferenceImage::GetCamera(CameraData& camera) const
{
  m_camera->GetData(camera);
}

std::shared_ptr<const Camera> ReferenceImage::GetCamera() const
{
  return m_camera;
}

std::shared_ptr<const Image> ReferenceImage::GetImage() const
{
  return m_image;
}

std::shared_ptr<const Image> ReferenceImage::GetMask() const
{
  return m_mask;
}

size_t ReferenceImage::GetValidPixelCount() const
{
  return m_validPixelCount;
}

void ReferenceImage::GetValidPixels(std::vector<uint2>& pixels) const
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

size_t ReferenceImage::GetValidPixelIndex(unsigned int x, unsigned int y) const
{
  const size_t index = y * m_mask->GetWidth() + x;
  const size_t validIndex = m_validPixelMap[index];
  const size_t totalPixels = m_mask->GetHeight() * m_mask->GetWidth();

  if (validIndex == totalPixels)
  {
    throw "pixel is not valid";
  }

  return validIndex;

  // float3* data = reinterpret_cast<float3*>(m_mask->GetData());
  // size_t count = 0;

  // for (unsigned int y = 0; y < m_mask->GetHeight(); ++y)
  // {
  //   for (unsigned int x = 0; x < m_mask->GetWidth(); ++x)
  //   {
  //     const size_t index = y * m_mask->GetWidth() + x;

  //     if (x == u && y == v)
  //     {
  //       if (data[index].x == 0)
  //         throw "pixel is not valid";

  //       return count;
  //     }

  //     if (data[index].x == 1) ++count;
  //   }
  // }

  // throw "pixel is not valid";
}

void ReferenceImage::Initialize()
{
  CreateImageMask();
  CreateValidPixelMap();
}

void ReferenceImage::CreateImageMask()
{
  m_mask = std::make_shared<Image>();
  m_camera->CaptureMask(*m_mask);
  m_mask->Save("temp.png");
}

void ReferenceImage::CreateValidPixelMap()
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