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
  m_sampleCount(1),
  m_detached(true)
{
  Initialize();
}

void Camera::SetImageSize(unsigned int w, unsigned int h)
{
  m_imageSize = make_uint2(w, h);
  m_buffer->setSize(m_imageSize.x, m_imageSize.y);
  m_context->MarkDirty();
}

void Camera::SetFocalLength(float fx, float fy)
{
  m_focalLength = make_float2(fx, fy);
  m_context->MarkDirty();
}

void Camera::SetCenterPoint(float cx, float cy)
{
  m_centerPoint = make_float2(cx, cy);
  m_context->MarkDirty();
}

void Camera::SetSampleCount(unsigned int count)
{
  m_sampleCount = count;
  m_program["sampleCount"]->setUint(m_sampleCount);
}

void Camera::Capture(Image& image)
{
  m_context->Launch(m_programId, m_imageSize);
  CopyBuffer(image);
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
  if (m_detached) UploadCamera(m_transform);
}

void Camera::CopyBuffer(Image& image)
{
  image.Resize(m_imageSize.x, m_imageSize.y);
  unsigned char* host = image.GetData();
  unsigned char* device = reinterpret_cast<unsigned char*>(m_buffer->map());
  std::copy(device, device + image.GetByteCount(), host);
  m_buffer->unmap();
}

void Camera::UploadCamera(const Transform& transform)
{
  const float cx = 1 - m_centerPoint.x / m_imageSize.x;
  const float cy = 1 - m_centerPoint.y / m_imageSize.y;

  const optix::Matrix3x4 matrix = transform.GetMatrix3x4();
  const float3 u = (m_imageSize.x / m_focalLength.x) * matrix.getCol(0);
  const float3 v = (m_imageSize.y / m_focalLength.y) * matrix.getCol(1);
  const float3 w = matrix.getCol(2);
  const float3 p = matrix.getCol(3);

  m_program["position"]->setFloat(p);
  m_program["center"]->setFloat(cx, cy);
  m_program["u"]->setFloat(u);
  m_program["v"]->setFloat(v);
  m_program["w"]->setFloat(w);
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
  m_program["sampleCount"]->setUint(m_sampleCount);
  m_program["buffer"]->setBuffer(m_buffer);
}

} // namespace torch