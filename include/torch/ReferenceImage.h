#pragma once

#include <torch/Core.h>

namespace torch
{

class ReferenceImage
{
  public:

    ReferenceImage(std::shared_ptr<Camera> camera,
        std::shared_ptr<Image> image);

    void GetCamera(CameraData& camera) const;

    std::shared_ptr<const Camera> GetCamera() const;

    std::shared_ptr<const Image> GetImage() const;

    std::shared_ptr<const Image> GetMask() const;

    size_t GetValidPixelCount() const;

    void GetValidPixels(std::vector<uint2>& pixels) const;

  private:

    void Initialize();

    void CreateImageMask();

  protected:

    std::shared_ptr<Camera> m_camera;

    std::shared_ptr<Image> m_image;

    std::shared_ptr<Image> m_mask;
};

} // namespace torch