#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/Node.h>
#include <torch/PtxUtil.h>

namespace torch
{

Camera::Camera(std::shared_ptr<Context> context) :
  Node(context),
  m_imageSize(make_uint2(1, 1)),
  m_focalLength(make_float2(0.5, 0.5)),
  m_centerPoint(make_float2(0.5, 0.5)),
  m_detached(true)
{
  Initialize();
}

Camera::~Camera()
{
}

void Camera::SetImageSize(unsigned int w, unsigned int h)
{
  m_imageSize = make_uint2(w, h);
  m_buffer->setSize(m_imageSize.x, m_imageSize.y);
  m_context->MarkDirty();
}

void Camera::SetFocalLength(float fx, float fy)
{
  m_centerPoint = make_float2(fx, fy);
  m_context->MarkDirty();
}

void Camera::SetCenterPoint(float cx, float cy)
{
  m_centerPoint = make_float2(cx, cy);
  m_context->MarkDirty();
}

void Camera::Capture()
{
  m_context->Launch(m_programId, m_imageSize);
}

void Camera::PreBuildScene()
{
  m_detached = true;
}

void Camera::BuildScene(Link& link)
{
  Node::BuildScene(link);
  UploadCamera(link.GetTransform() * m_transform);
  m_detached = false;
}

void Camera::PostBuildScene()
{
  UploadCamera(m_transform);
}

void Camera::UploadCamera(const Transform& transform)
{
}

void Camera::Initialize()
{
  CreateBuffer();
  CreateProgram();
}

void Camera::CreateBuffer()
{
  m_buffer = m_context->CreateBuffer(RT_BUFFER_OUTPUT);
  m_buffer->setFormat(RT_FORMAT_FLOAT3);
  m_buffer->setSize(m_imageSize.x, m_imageSize.y);
}

void Camera::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("Camera");
  m_program = m_context->CreateProgram(file, "Capture");
  m_programId = m_context->RegisterLaunchProgram(m_program);
  m_program["buffer"]->setBuffer(m_buffer);
}

} // namespace torch