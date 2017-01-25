#pragma once

#include <optixu/optixpp.h>
#include <torch/Node.h>

namespace torch
{

class Camera : public Node
{
  public:

    Camera(std::shared_ptr<Context> context);

    ~Camera();

    void SetImageSize(unsigned int w, unsigned int h);

    void SetFocalLength(float fx, float fy);

    void SetCenterPoint(float cx, float cy);

    void Capture();

    void PreBuildScene() override;

    void BuildScene(Link& link) override;

    void PostBuildScene() override;

  protected:

    void UploadCamera(const Transform& transform);

  private:

    void Initialize();

    void CreateBuffer();

    void CreateProgram();

  protected:

    uint2 m_imageSize;

    float2 m_focalLength;

    float2 m_centerPoint;

    optix::Buffer m_buffer;

    optix::Program m_program;

    unsigned int m_programId;

    bool m_detached;
};

} // namespace torch