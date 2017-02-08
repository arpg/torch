#pragma once

#include <torch/Node.h>

namespace torch
{

class Camera : public Node
{
  public:

    Camera(std::shared_ptr<Context> context);

    void GetImageSize(unsigned int& w, unsigned int& h) const;

    void SetImageSize(unsigned int w, unsigned int h);

    void SetFocalLength(float fx, float fy);

    void SetCenterPoint(float cx, float cy);

    void SetSampleCount(unsigned int count);

    void Capture(Image& image);

    void CaptureMask(Image& image);

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

    void CreateDepthBuffer();

    void CreateDepthProgram();

    void CreateMaskProgram();

  protected:

    uint2 m_imageSize;

    float2 m_focalLength;

    float2 m_centerPoint;

    unsigned int m_sampleCount;

    optix::Buffer m_buffer;

    optix::Buffer m_depthBuffer;

    optix::Program m_program;

    unsigned int m_programId;

    optix::Program m_depthProgram;

    unsigned int m_depthProgramId;

    optix::Program m_maskProgram;

    unsigned int m_maskProgramId;

    bool m_detached;
};

} // namespace torch