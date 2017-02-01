#pragma once

#include <torch/Node.h>

namespace torch
{

class Camera : public Node
{
  public:

    Camera(std::shared_ptr<Context> context);

    void SetImageSize(unsigned int w, unsigned int h);

    void SetFocalLength(float fx, float fy);

    void SetCenterPoint(float cx, float cy);

    void SetSampleCount(unsigned int count);

    void Capture(Image& image);

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  protected:

    void CopyBuffer(Image& image);

    void UploadCamera(const Transform& transform);

    void GetData(const Transform& transform, CameraData& data) const;

  private:

    void Initialize();

    void CreateBuffer();

    void CreateProgram();

  protected:

    uint2 m_imageSize;

    float2 m_focalLength;

    float2 m_centerPoint;

    unsigned int m_sampleCount;

    optix::Buffer m_buffer;

    optix::Program m_program;

    unsigned int m_programId;

    bool m_detached;
};

} // namespace torch