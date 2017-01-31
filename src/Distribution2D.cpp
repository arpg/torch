#include <torch/Distribution2D.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

#include <iostream>

namespace torch
{

Distribution2D::Distribution2D(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

optix::Program Distribution2D::GetProgram() const
{
  return m_program;
}

void Distribution2D::SetValues(const std::vector<float>& values,
    const std::vector<unsigned int>& offsets)
{
  const size_t rowCount = offsets.size() - 1;
  std::vector<float> rowCdf(rowCount);
  std::vector<float> colCdfs(values.size());
  size_t index = 0;

  for (size_t row = 0; row < rowCount; ++row)
  {
    const size_t colCount = offsets[row + 1] - offsets[row];

    for (size_t col = 0; col < colCount; ++col)
    {
      const size_t i = index + col;
      const float value = values[i];
      colCdfs[i] = (col == 0) ? value : value + colCdfs[i - 1];
    }

    const float integral = colCdfs[index + colCount - 1];

    for (size_t col = 0; col < colCount; ++col)
    {
      colCdfs[index++] /= integral;
    }

    rowCdf[row] = (row == 0) ? integral : integral + rowCdf[row - 1];
  }

  const float integral = rowCdf[rowCount - 1];

  for (size_t row = 0; row < rowCount; ++row)
  {
    rowCdf[row] /= integral;
  }

  CopyVector(rowCdf, m_rowCdf);
  CopyVector(colCdfs, m_colCdfs);
  CopyVector(offsets, m_offsets);
}

template<typename T>
void Distribution2D::CopyVector(const std::vector<T>& values,
    optix::Buffer buffer)
{
  buffer->setSize(values.size());
  T* device = reinterpret_cast<T*>(buffer->map());
  std::copy(values.begin(), values.end(), device);
  buffer->unmap();
}

void Distribution2D::Initialize()
{
  CreateProgram();
  CreateRowBuffer();
  CreateColumnBuffer();
  CreateOffsetBuffer();
}

void Distribution2D::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("Distribution2D");
  m_program = m_context->CreateProgram(file, "Sample");
}

void Distribution2D::CreateRowBuffer()
{
  m_rowCdf = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["rowCdf"]->setBuffer(m_rowCdf);
  m_rowCdf->setFormat(RT_FORMAT_FLOAT);
  m_rowCdf->setSize(0);
}

void Distribution2D::CreateColumnBuffer()
{
  m_colCdfs = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["colCdfs"]->setBuffer(m_colCdfs);
  m_colCdfs->setFormat(RT_FORMAT_FLOAT);
  m_colCdfs->setSize(0);
}

void Distribution2D::CreateOffsetBuffer()
{
  m_offsets = m_context->CreateBuffer(RT_BUFFER_INPUT);
  m_program["offsets"]->setBuffer(m_offsets);
  m_offsets->setFormat(RT_FORMAT_UNSIGNED_INT);
  m_offsets->setSize(0);
}

} // namespace torch