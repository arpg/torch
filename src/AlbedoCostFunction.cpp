#include <torch/AlbedoCostFunction.h>
#include <lynx/lynx.h>
#include <torch/Context.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/Keyframe.h>
#include <torch/KeyframeSet.h>
#include <torch/PtxUtil.h>
#include <torch/SparseMatrix.h>

#include <torch/Image.h>
#include <iostream>

namespace torch
{

AlbedoCostFunction::AlbedoCostFunction(std::shared_ptr<MatteMaterial> material,
    std::shared_ptr<Mesh> mesh) :
  m_locked(false),
  m_material(material),
  m_mesh(mesh),
  m_jacobian(nullptr),
  m_jacobianValues(nullptr),
  m_referenceValues(nullptr),
  m_valueCount(0),
  m_rowIndices(nullptr),
  m_colIndices(nullptr),
  m_dirty(true)
{
  Initialize();
}

AlbedoCostFunction::~AlbedoCostFunction()
{
  cudaFree(m_referenceValues);
  cudaFree(m_rowIndices);
  cudaFree(m_colIndices);
}

void AlbedoCostFunction::AddKeyframe(std::shared_ptr<Keyframe> keyframe)
{
  LYNX_ASSERT(!m_locked, "new keyframes cannot be added");
  const size_t residualCount = 3 * keyframe->GetValidPixelCount();
  lynx::CostFunction::m_residualCount += residualCount;
  lynx::CostFunction::m_maxEvaluationBlockSize += residualCount;
  m_keyframes->Add(keyframe);
  m_referenceValues = nullptr;
  m_jacobianValues = nullptr;
  m_dirty = true;
}

lynx::Matrix* AlbedoCostFunction::CreateJacobianMatrix()
{
  m_locked = true;
  PrepareEvaluation();
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;

  return new lynx::SparseMatrix3C(m_jacobianValues, m_valueCount, m_rowIndices,
      m_colIndices, rows, cols);
}

void AlbedoCostFunction::Evaluate(const float* const* parameters,
    float* residuals)
{
  PrepareEvaluation();
  const size_t rows = GetResidualCount() / 3;
  const size_t cols = GetParameterCount() / 3;

  lynx::SparseMatrix3C jacobian(m_jacobianValues, m_valueCount, m_rowIndices,
      m_colIndices, rows, cols);

  jacobian.RightMultiply(parameters[0], residuals);

  // // std::cout << "Render Sum:    " << lynx::Sum(residuals, GetResidualCount()) << std::endl;
  // // std::cout << "Reference Sum: " << lynx::Sum(m_referenceValues, GetResidualCount()) << std::endl;

  // std::vector<float3> values(GetResidualCount() / 3);
  // const size_t bytes = sizeof(float) * GetResidualCount();
  // const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  // LYNX_CHECK_CUDA(cudaMemcpy(values.data(), residuals, bytes, kind));
  // // LYNX_CHECK_CUDA(cudaMemcpy(values.data(), m_referenceValues, bytes, kind));

  // std::vector<uint2> pixels;
  // m_keyframes->Get(0)->GetValidPixels(pixels);

  // Image image;
  // image.Resize(160, 120);
  // float3* data = reinterpret_cast<float3*>(image.GetData());
  // std::fill(data, data + 160 * 120, make_float3(0, 0, 0));

  // for (size_t i = 0; i < values.size(); ++i)
  // {
  //   const uint2& pixel = pixels[i];
  //   const size_t index = pixel.y * 160 + pixel.x;
  //   values[i].x = fmaxf(fminf(fabs(values[i].x), 1.0f), 0.0f);
  //   values[i].y = fmaxf(fminf(fabs(values[i].y), 1.0f), 0.0f);
  //   values[i].z = fmaxf(fminf(fabs(values[i].z), 1.0f), 0.0f);
  //   // if (values[i].x > 0) std::cout << "Pixel: " << values[i].x << std::endl;
  //   data[index] = values[i];
  // }

  // image.Save("render.png");
  // // std::exit(0);

  lynx::Add(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::Evaluate(size_t offset, size_t size,
    const float* const* parameters, float* residuals, lynx::Matrix* jacobian)
{
  LYNX_ASSERT(jacobian->GetValues() == m_jacobianValues, "invalid jacobian");
  LYNX_ASSERT(offset == 0, "sub-problem cannot be evaluated");
  LYNX_ASSERT(size == GetResidualCount(), "sub-problem cannot be evaluated");

  PrepareEvaluation();
  jacobian->RightMultiply(parameters[0], residuals);
  lynx::Add(m_referenceValues, residuals, residuals, GetResidualCount());
}

void AlbedoCostFunction::ClearJacobian()
{
  m_jacobianValues = nullptr;
  m_referenceValues = nullptr;
}

void AlbedoCostFunction::PrepareEvaluation()
{
  m_keyframes->UpdateBuffers();

  if (m_dirty)
  {
    CreatePixelVertexBuffer();
    m_dirty = false;
  }

  if (!m_jacobianValues)
  {
    ComputeJacobian();
  }

  if (!m_referenceValues)
  {
    CreateReferenceBuffer();
  }
}

void AlbedoCostFunction::CreatePixelVertexBuffer()
{
  const size_t imageCount = m_keyframes->Size();
  const size_t vertexCount = m_mesh->GetVertexCount();
  const size_t launchSize = imageCount * vertexCount;
  std::shared_ptr<Context> context = m_material->GetContext();

  m_boundingBoxes->setSize(launchSize);
  context->Launch(m_boundsProgramId, launchSize);

  std::vector<uint4> boundingBoxes(launchSize);
  uint4* device = reinterpret_cast<uint4*>(m_boundingBoxes->map());
  std::copy(device, device + launchSize, boundingBoxes.data());
  m_boundingBoxes->unmap();

  const size_t validPixelCount = m_keyframes->GetValidPixelCount();
  std::vector<std::vector<unsigned int>> pixelVertexMap(validPixelCount);
  size_t totalCount = 0;
  size_t pixelOffset = 0;

  for (size_t i = 0; i < imageCount; ++i)
  {
    const size_t offset = i * vertexCount;
    std::shared_ptr<Keyframe> keyframe = m_keyframes->Get(i);

    for (size_t j = 0; j < vertexCount; ++j)
    {
      const uint4& box = boundingBoxes[offset + j];

      for (unsigned int y = box.y; y <= box.w; ++y)
      {
        for (unsigned int x = box.x; x <= box.z; ++x)
        {
          if (keyframe->IsValidPixel(x, y))
          {
            const size_t index = pixelOffset + keyframe->GetValidPixelIndex(x, y);
            pixelVertexMap[index].push_back(j);
            ++totalCount;
          }
        }
      }
    }

    pixelOffset += keyframe->GetValidPixelCount();
  }

  size_t mapIndex = 0;
  std::vector<unsigned int> rowIndices(totalCount);
  std::vector<unsigned int> colIndices(totalCount);
  std::vector<unsigned int> offsets(pixelVertexMap.size() + 1);
  offsets[0] = 0;

  for (size_t i = 0; i < pixelVertexMap.size(); ++i)
  {
    for (unsigned int j : pixelVertexMap[i])
    {
      rowIndices[mapIndex] = i;
      colIndices[mapIndex] = j;
      ++mapIndex;
    }

    offsets[i + 1] = pixelVertexMap[i].size() + offsets[i];
  }

  m_jacobian->Allocate(offsets, colIndices);

  optix::Buffer values = m_jacobian->GetValuesBuffer();
  float* deviceValues = reinterpret_cast<float*>(values->map());
  std::fill(deviceValues, deviceValues + (3 * totalCount), 0.0f);
  values->unmap();

  m_valueCount = totalCount;

  const size_t bytes = sizeof(float) * totalCount;
  LYNX_CHECK_CUDA(cudaMalloc(&m_rowIndices, bytes));
  LYNX_CHECK_CUDA(cudaMalloc(&m_colIndices, bytes));

  LYNX_CHECK_CUDA(cudaMemcpy(m_rowIndices, rowIndices.data(), bytes,
                             cudaMemcpyHostToDevice));

  LYNX_CHECK_CUDA(cudaMemcpy(m_colIndices, colIndices.data(), bytes,
                             cudaMemcpyHostToDevice));
}

void AlbedoCostFunction::ComputeJacobian()
{
  LYNX_ASSERT(!m_keyframes->Empty(), "no keyframes have been added");

  m_jacobian->SetZero();

  const size_t launchSize = m_keyframes->GetValidPixelCount();
  std::shared_ptr<Context> context = m_material->GetContext();
  context->GetVariable("computeAlbedoDerivs")->setUint(true);
  context->Launch(m_captureProgramId, launchSize);
  context->GetVariable("computeAlbedoDerivs")->setUint(false);

  optix::Buffer buffer = m_jacobian->GetValuesBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  m_jacobianValues = reinterpret_cast<float*>(pointer);
}

void AlbedoCostFunction::CreateReferenceBuffer()
{
  optix::Buffer reference = m_keyframes->GetReferenceBuffer();
  CUdeviceptr referencePtr = reference->getDevicePointer(0);
  float* referenceValues = reinterpret_cast<float*>(referencePtr);

  optix::Buffer render = m_keyframes->GetRenderBuffer();
  CUdeviceptr renderPtr = render->getDevicePointer(0);
  float* renderValues = reinterpret_cast<float*>(renderPtr);

  const size_t count = 3 * m_keyframes->GetValidPixelCount();
  const size_t bytes = sizeof(float) * count;
  LYNX_CHECK_CUDA(cudaMalloc(&m_referenceValues, bytes));
  lynx::Sub(referenceValues, renderValues, m_referenceValues, count);

  // optix::Buffer reference = m_keyframes->GetReferenceBuffer();
  // CUdeviceptr referencePtr = reference->getDevicePointer(0);
  // m_referenceValues = reinterpret_cast<float*>(referencePtr);
}

void AlbedoCostFunction::Initialize()
{
  SetDimensions();
  CreateJacobianBuffer();
  CreateBoundingBoxBuffer();
  CreateAdjacencyBuffers();
  CreateCaptureProgram();
  CreateBoundsProgram();
  CreateKeyframeSet();
}

void AlbedoCostFunction::SetDimensions()
{
  const size_t paramCount = 3 * m_material->GetAlbedoCount();
  lynx::CostFunction::m_parameterBlockSizes.push_back(paramCount);
  lynx::CostFunction::m_maxEvaluationBlockSize = 0;
  lynx::CostFunction::m_residualCount = 0;
}

void AlbedoCostFunction::CreateJacobianBuffer()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  m_jacobian = std::make_shared<SparseMatrix>(context);
  m_material->SetDerivativeProgram(m_jacobian->GetAddProgram());
}

void AlbedoCostFunction::CreateBoundingBoxBuffer()
{
  std::shared_ptr<Context> context;
  context = m_material->GetContext();
  m_boundingBoxes = context->CreateBuffer(RT_BUFFER_OUTPUT);
  m_boundingBoxes->setFormat(RT_FORMAT_UNSIGNED_INT4);
  m_boundingBoxes->setSize(0);
}

void AlbedoCostFunction::CreateAdjacencyBuffers()
{
  unsigned int* device;
  std::shared_ptr<Context> context;
  context = m_material->GetContext();
  std::vector<unsigned int> indices;
  std::vector<unsigned int> offsets;
  m_mesh->GetVertexAdjacencyMap(indices, offsets, false);

  m_neighborIndices = context->CreateBuffer(RT_BUFFER_INPUT);
  m_neighborIndices->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_neighborIndices->setSize(indices.size());

  m_neighborOffsets = context->CreateBuffer(RT_BUFFER_INPUT);
  m_neighborOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_neighborOffsets->setSize(offsets.size());

  device = reinterpret_cast<unsigned int*>(m_neighborIndices->map());
  std::copy(indices.begin(), indices.end(), device);
  m_neighborIndices->unmap();

  device = reinterpret_cast<unsigned int*>(m_neighborOffsets->map());
  std::copy(offsets.begin(), offsets.end(), device);
  m_neighborOffsets->unmap();
}

void AlbedoCostFunction::CreateCaptureProgram()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  const std::string file = PtxUtil::GetFile("AlbedoCostFunction");
  m_captureProgram = context->CreateProgram(file, "Capture");
  m_captureProgramId = context->RegisterLaunchProgram(m_captureProgram);
  m_captureProgram["AddToAlbedoJacobian"]->setProgramId(m_jacobian->GetAddProgram());
}

void AlbedoCostFunction::CreateBoundsProgram()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  const std::string file = PtxUtil::GetFile("AlbedoCostFunction");
  m_boundsProgram = context->CreateProgram(file, "GetBoundingBoxes");
  m_boundsProgramId = context->RegisterLaunchProgram(m_boundsProgram);
  m_boundsProgram["neighborIndices"]->setBuffer(m_neighborIndices);
  m_boundsProgram["neighborOffsets"]->setBuffer(m_neighborOffsets);
  m_boundsProgram["boundingBoxes"]->setBuffer(m_boundingBoxes);
  m_boundsProgram["vertices"]->setBuffer(m_mesh->GetVertexBuffer());

}

void AlbedoCostFunction::CreateKeyframeSet()
{
  std::shared_ptr<Context> context = m_material->GetContext();
  m_keyframes = std::make_unique<KeyframeSet>(context);
  m_captureProgram["cameras"]->setBuffer(m_keyframes->GetCameraBuffer());
  m_captureProgram["pixelSamples"]->setBuffer(m_keyframes->GetPixelBuffer());
  m_captureProgram["render"]->setBuffer(m_keyframes->GetRenderBuffer());
  m_boundsProgram["cameras"]->setBuffer(m_keyframes->GetCameraBuffer());
  m_material->SetCameraBuffer(m_keyframes->GetCameraBuffer());
  m_material->SetPixelBuffer(m_keyframes->GetPixelBuffer());
}

} // namespace torch