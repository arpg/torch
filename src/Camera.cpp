#include <torch/Camera.h>
#include <torch/Context.h>
#include <torch/Image.h>
#include <torch/Link.h>
#include <torch/Node.h>
#include <torch/PtxUtil.h>
#include <torch/device/Camera.h>

namespace torch
{

Camera::Camera(std::shared_ptr<Context> context) :
  Node(context),
  m_imageSize(make_uint2(1, 1)),
  m_focalLength(make_float2(0.5, 0.5)),
  m_centerPoint(make_float2(0.5, 0.5)),
  m_sampleCount(1),
  m_maxDepth(6),
  m_detached(true)
{
  Initialize();
}

void Camera::GetImageSize(unsigned int& w, unsigned int& h) const
{
  w = m_imageSize.x;
  h = m_imageSize.y;
}

void Camera::SetImageSize(unsigned int w, unsigned int h)
{
  m_imageSize = make_uint2(w, h);
  m_buffer->setSize(m_imageSize.x, m_imageSize.y);
  m_depthBuffer->setSize(m_imageSize.x, m_imageSize.y);
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

unsigned int Camera::GetSampleCount() const
{
  return m_sampleCount;
}

void Camera::SetSampleCount(unsigned int count)
{
  m_sampleCount = count;
  m_program["sampleCount"]->setUint(m_sampleCount);
  m_depthProgram["sampleCount"]->setUint(m_sampleCount);
  m_maskProgram["sampleCount"]->setUint(m_sampleCount);
}

unsigned int Camera::GetMaxDepth() const
{
  return m_maxDepth;
}

void Camera::SetMaxDepth(unsigned int depth)
{
  m_maxDepth = depth;
  m_program["maxDepth"]->setUint(m_maxDepth);
  m_depthProgram["maxDepth"]->setUint(m_maxDepth);
  m_maskProgram["maxDepth"]->setUint(m_maxDepth);
}

void Camera::Capture(Image& image)
{
  m_context->Launch(m_programId, m_imageSize);
  CopyBuffer(image);
}

void Camera::CaptureAlbedo(Image& image)
{
  m_context->GetVariable("albedoOnly")->setUint(true);
  m_context->Launch(m_programId, m_imageSize);
  m_context->GetVariable("albedoOnly")->setUint(false);
  CopyBuffer(image);
}

void Camera::CaptureLighting(Image& image)
{
  m_context->GetVariable("lightingOnly")->setUint(true);
  m_context->Launch(m_programId, m_imageSize);
  m_context->GetVariable("lightingOnly")->setUint(false);
  CopyBuffer(image);
}

void Camera::CaptureMask(Image& image)
{
  m_context->Launch(m_depthProgramId, m_imageSize);
  m_context->Launch(m_maskProgramId, m_imageSize);
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

void Camera::GetData(CameraData& data) const
{
  return GetData(m_transform, data);
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
  CameraData data;
  GetData(transform, data);
  m_program["camera"]->setUserData(sizeof(CameraData), &data);
  m_depthProgram["camera"]->setUserData(sizeof(CameraData), &data);
  m_maskProgram["camera"]->setUserData(sizeof(CameraData), &data);
}

void Camera::GetData(const Transform& transform, CameraData& data) const
{
  const optix::Matrix3x4 matrix = transform.GetMatrix3x4();
  data.center.x = 1 - m_centerPoint.x / m_imageSize.x;
  data.center.y = 1 - m_centerPoint.y / m_imageSize.y;
  data.u = (m_imageSize.x / m_focalLength.x) * matrix.getCol(0);
  data.v = (m_imageSize.y / m_focalLength.y) * matrix.getCol(1);
  data.w = matrix.getCol(2);
  data.position = matrix.getCol(3);
  data.samples = m_sampleCount;
  data.maxDepth = m_maxDepth;
  data.imageSize = m_imageSize;

  data.K.setRow(0, make_float3(m_focalLength.x, 0, m_centerPoint.x));
  data.K.setRow(1, make_float3(0, m_focalLength.y, m_centerPoint.y));
  data.K.setRow(2, make_float3(0, 0, 1));

  data.Kinv.setRow(0, make_float3(1 / m_focalLength.x, 0, -m_centerPoint.x / m_focalLength.x));
  data.Kinv.setRow(1, make_float3(0, 1 / m_focalLength.y, -m_centerPoint.y / m_focalLength.x));
  data.Kinv.setRow(2, make_float3(0, 0, 1));

  data.Twc = transform.GetMatrix();
  data.Tcw = transform.Inverse().GetMatrix();
}

void Camera::Initialize()
{
  CreateBuffer();
  CreateProgram();
  CreateDepthBuffer();
  CreateDepthProgram();
  CreateMaskProgram();
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
  m_program["maxDepth"]->setUint(m_maxDepth);
  m_program["buffer"]->setBuffer(m_buffer);
}

void Camera::CreateDepthBuffer()
{
  m_depthBuffer = m_context->CreateBuffer(RT_BUFFER_OUTPUT);
  m_depthBuffer->setFormat(RT_FORMAT_FLOAT);
  m_depthBuffer->setSize(m_imageSize.x, m_imageSize.y);
}

void Camera::CreateDepthProgram()
{
  const std::string file = PtxUtil::GetFile("Camera");
  m_depthProgram = m_context->CreateProgram(file, "CaptureDepth");
  m_depthProgramId = m_context->RegisterLaunchProgram(m_depthProgram);
  m_depthProgram["sampleCount"]->setUint(m_sampleCount);
  m_depthProgram["maxDepth"]->setUint(m_maxDepth);
  m_depthProgram["depthBuffer"]->setBuffer(m_depthBuffer);
  m_depthProgram["buffer"]->setBuffer(m_buffer);
}

void Camera::CreateMaskProgram()
{
  const std::string file = PtxUtil::GetFile("Camera");
  m_maskProgram = m_context->CreateProgram(file, "CaptureMask");
  m_maskProgramId = m_context->RegisterLaunchProgram(m_maskProgram);
  m_maskProgram["sampleCount"]->setUint(m_sampleCount);
  m_maskProgram["maxDepth"]->setUint(m_maxDepth);
  m_maskProgram["depthBuffer"]->setBuffer(m_depthBuffer);
  m_maskProgram["buffer"]->setBuffer(m_buffer);
}

} // namespace torch
