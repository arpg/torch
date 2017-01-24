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
};

} // namespace torch