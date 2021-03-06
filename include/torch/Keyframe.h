#pragma once

#include <torch/Core.h>

namespace torch
{

class Keyframe
{
  public:

    Keyframe(std::shared_ptr<Camera> camera,
        std::shared_ptr<Image> image);

    void GetCamera(CameraData& camera) const;

    std::shared_ptr<const Camera> GetCamera() const;

    std::shared_ptr<const Image> GetImage() const;

    std::shared_ptr<const Image> GetMask() const;

    size_t GetValidPixelCount() const;

    void GetValidPixels(std::vector<uint2>& pixels) const;

    size_t GetValidPixelIndex(unsigned int x, unsigned int y) const;

    void GetValidPixelRadiance(std::vector<Spectrum>& radiance);

    bool IsValidPixel(unsigned int x, unsigned int y) const;

  private:

    void Initialize();

    void CreateImageMask();

    void CreateValidPixelMap();

  protected:

    std::shared_ptr<Camera> m_camera;

    std::shared_ptr<Image> m_image;

    std::shared_ptr<Image> m_mask;

    std::vector<size_t> m_validPixelMap;

    size_t m_validPixelCount;
};

} // namespace torch