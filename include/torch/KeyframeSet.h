#pragma once

#include <torch/Core.h>

namespace torch
{

class KeyframeSet
{
  public:

    KeyframeSet(std::shared_ptr<Context> context);

    ~KeyframeSet();

    bool Empty() const;

    void Add(std::shared_ptr<Keyframe> keyframe);

    size_t GetValidPixelCount() const;

    optix::Buffer GetCameraBuffer();

    optix::Buffer GetPixelBuffer();

    optix::Buffer GetReferenceBuffer();

    optix::Buffer GetRenderBuffer();

    void UpdateBuffers();

  protected:

    void UpdateCameraBuffer();

    void UpdatePixelBuffer();

    void UpdateReferenceBuffer();

    void UpdateRenderBuffer();

  private:

    void Initialize();

    void CreateCameraBuffer();

    void CreatePixelBuffer();

    void CreateReferenceBuffer();

    void CreateRenderBuffer();

  protected:

    std::vector<std::shared_ptr<Keyframe>> m_keyframes;

    std::shared_ptr<Context> m_context;

    optix::Buffer m_cameras;

    optix::Buffer m_pixels;

    optix::Buffer m_reference;

    optix::Buffer m_render;

    size_t m_validPixelCount;

    bool m_dirty;
};

} // namespace torch