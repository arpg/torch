#include <torch/Distribution2D.h>
#include <torch/Context.h>
#include <torch/PtxUtil.h>

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

void Distribution2D::SetValues(const std::vector<std::vector<float>>& values)
{
  std::vector<unsigned int> offsets(values.size() + 1);
  offsets[0] = 0;

  for (unsigned int i = 0; i < values.size(); ++i)
  {
    offsets[i + 1] = values[i].size() + offsets[i];
  }

  std::vector<float> colCdfs(offsets.back());
  std::vector<float> rowCdf(values.size());

  for (unsigned int i = 0; i < values.size(); ++i)
  {
    const unsigned int offset = offsets[i];
    float previous = 0;

    for (unsigned int j = 0; j < values[i].size(); ++j)
    {
      colCdfs[offset + j] = values[i][j] + previous;
      previous = colCdfs[offset + j];
    }

    rowCdf[i] = colCdfs[offsets[i + 1] - 1];

    for (unsigned int j = 0; j < values[i].size(); ++j)
    {
      colCdfs[offset + j] /= rowCdf[i];
    }
  }

  const float integral = rowCdf.back();

  for (unsigned int i = 0; i < values.size(); ++i)
  {
    rowCdf[i] /= integral;
  }
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