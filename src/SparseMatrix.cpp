#include <torch/SparseMatrix.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

namespace torch
{

SparseMatrix::SparseMatrix(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

void SparseMatrix::Allocate(const std::vector<unsigned int>& rowOffsets,
    const std::vector<unsigned int>& colIndices)
{
  AllocateValues(colIndices.size());
  SetRowOffsets(rowOffsets);
  SetColumnIndices(colIndices);
}

optix::Program SparseMatrix::GetAddProgram() const
{
  return m_program;
}

void SparseMatrix::GetValues(std::vector<float3>& values)
{
  RTsize size;
  m_values->getSize(size);
  values.resize(size);

  float3* device = reinterpret_cast<float3*>(m_values->map());
  std::copy(device, device + size, values.data());
  m_values->unmap();
}

void SparseMatrix::GetRowOffsets(std::vector<unsigned int>& offsets)
{
  RTsize size;
  m_rowOffsets->getSize(size);
  offsets.resize(size);

  unsigned int* device = reinterpret_cast<unsigned int*>(m_rowOffsets->map());
  std::copy(device, device + size, offsets.data());
  m_rowOffsets->unmap();
}

void SparseMatrix::GetColumnIndices(std::vector<unsigned int>& indices)
{
  RTsize size;
  m_colIndices->getSize(size);
  indices.resize(size);

  unsigned int* device = reinterpret_cast<unsigned int*>(m_colIndices->map());
  std::copy(device, device + size, indices.data());
  m_colIndices->unmap();
}

void SparseMatrix::AllocateValues(size_t size)
{
  m_values->setSize(size);
  float3* device = reinterpret_cast<float3*>(m_values->map());
  std::fill(device, device + size, make_float3(0, 0, 0));
  m_values->unmap();
}

void SparseMatrix::SetRowOffsets(const std::vector<unsigned int>& offsets)
{
  m_rowOffsets->setSize(offsets.size());
  unsigned int* device = reinterpret_cast<unsigned int*>(m_rowOffsets->map());
  std::copy(offsets.begin(), offsets.end(), device);
  m_rowOffsets->unmap();
}

void SparseMatrix::SetColumnIndices(const std::vector<unsigned int>& indices)
{
  m_colIndices->setSize(indices.size());
  unsigned int* device = reinterpret_cast<unsigned int*>(m_colIndices->map());
  std::copy(indices.begin(), indices.end(), device);
  m_colIndices->unmap();
}

void SparseMatrix::Initialize()
{
  CreateValueBuffer();
  CreateRowOffsetBuffer();
  CreateColumnIndexBuffer();
  CreateProgram();
}

void SparseMatrix::CreateValueBuffer()
{
  m_values = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_values->setFormat(RT_FORMAT_FLOAT3);
  m_values->setSize(0);
}

void SparseMatrix::CreateRowOffsetBuffer()
{
  m_rowOffsets = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_rowOffsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_rowOffsets->setSize(0);
}

void SparseMatrix::CreateColumnIndexBuffer()
{
  m_colIndices = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
  m_colIndices->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_colIndices->setSize(0);
}

void SparseMatrix::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("SparseMatrix");
  m_program = m_context->CreateProgram(file, "Add");
  m_program["values"]->setBuffer(m_values);
  m_program["rowOffsets"]->setBuffer(m_rowOffsets);
  m_program["colIndices"]->setBuffer(m_colIndices);
}

} // namespace