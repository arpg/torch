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

    size_t Size() const;

    void Add(std::shared_ptr<Keyframe> keyframe);

    std::shared_ptr<Keyframe> operator[](size_t index) const;

    std::shared_ptr<Keyframe> Get(size_t index) const;

    size_t GetValidPixelIndex(size_t index, unsigned int x, unsigned int y) const;

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