#include <gtest/gtest.h>
#include <torch/Torch.h>
#include "TestPtxUtil.h"

struct Distribution1D : public testing::Test
{
  Distribution1D() :
    m_sampleCount(0)
  {
    Initialize();
  }

  void SetValues(const std::vector<float>& values)
  {
    m_distribution->SetValues(values);
    m_countBuffer->setSize(values.size());
    m_pdfBuffer->setSize(values.size());
    SetSampleCount();
  }

  void Sample(std::vector<float>& ratios, std::vector<float>& pdfs)
  {
    ClearBuffers();
    LaunchProgram();
    GetSampleRatios(ratios);
    GetSamplePdfs(pdfs);
  }

  void SetSampleCount()
  {
    m_sampleCount = 1E4 * GetBufferSize(m_countBuffer) / epsilon;
  }

  void ClearBuffers()
  {
    FillBuffer(m_countBuffer, 0u);
    FillBuffer(m_pdfBuffer, 0.0f);
  }

  void LaunchProgram()
  {
    m_context->Launch(m_programId, m_sampleCount);
  }

  void GetSampleRatios(std::vector<float>& ratios)
  {
    const RTsize size = GetBufferSize(m_countBuffer);

    ratios.resize(size);
    std::vector<unsigned int> counts(size);
    CopyBuffer(m_countBuffer, counts);

    for (size_t i = 0; i < counts.size(); ++i)
    {
      ratios[i] = float(counts[i]) / m_sampleCount;
    }
  }

  void GetSamplePdfs(std::vector<float>& pdfs)
  {
    CopyBuffer(m_pdfBuffer, pdfs);
  }

  static void Normalize(std::vector<float>& values)
  {
    float sum = 0;
    for (float value : values) sum += value;
    for (size_t i = 0; i < values.size(); ++i) values[i] /= sum;
  }

  static RTsize GetBufferSize(optix::Buffer buffer)
  {
    RTsize w, h, d;
    buffer->getSize(w, h, d);
    return w * h * d;
  }

  template <typename T>
  void FillBuffer(optix::Buffer buffer, T value)
  {
    T* device = reinterpret_cast<T*>(buffer->map());
    std::fill(device, device + GetBufferSize(buffer), value);
    buffer->unmap();
  }

  template <typename T>
  void CopyBuffer(optix::Buffer buffer, std::vector<T>& values)
  {
    values.resize(GetBufferSize(buffer));
    T* device = reinterpret_cast<T*>(buffer->map());
    std::copy(device, device + values.size(), values.data());
    buffer->unmap();
  }

  void Initialize()
  {
    CreateContext();
    CreateProgram();
    CreateCountBuffer();
    CreatePdfBuffer();
    CreateDistribution();
  }

  void CreateContext()
  {
    m_context = torch::Context::Create();
  }

  void CreateProgram()
  {
    const std::string file = torch::TestPtxUtil::GetFile("Distribution1D");
    m_program = m_context->CreateProgram(file, "Sample");
    m_programId = m_context->RegisterLaunchProgram(m_program);
  }

  void CreateCountBuffer()
  {
    m_countBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
    m_countBuffer->setFormat(RT_FORMAT_UNSIGNED_INT);
    m_countBuffer->setSize(0);
    m_program["counts"]->setBuffer(m_countBuffer);
  }

  void CreatePdfBuffer()
  {
    m_pdfBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
    m_pdfBuffer->setFormat(RT_FORMAT_FLOAT);
    m_pdfBuffer->setSize(0);
    m_program["pdfs"]->setBuffer(m_pdfBuffer);
  }

  void CreateDistribution()
  {
    m_distribution = std::make_unique<torch::Distribution1D>(m_context);
    m_program["Sample1D"]->set(m_distribution->GetProgram());
  }

  unsigned int m_sampleCount;

  unsigned int m_programId;

  optix::Program m_program;

  optix::Buffer m_countBuffer;

  optix::Buffer m_pdfBuffer;

  std::shared_ptr<torch::Context> m_context;

  std::unique_ptr<torch::Distribution1D> m_distribution;

  static constexpr float epsilon = 1E-4f;
};

TEST_F(Distribution1D, Normalized1)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(1.0f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Normalized2)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(0.3f);
  expected.push_back(0.7f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Normalized3)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(0.8f);
  expected.push_back(0.0f);
  expected.push_back(0.2f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Normalized4)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(0.1f);
  expected.push_back(0.2f);
  expected.push_back(0.3f);
  expected.push_back(0.4f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}


TEST_F(Distribution1D, Normalized5)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(0.2f);
  expected.push_back(0.2f);
  expected.push_back(0.2f);
  expected.push_back(0.2f);
  expected.push_back(0.2f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}


TEST_F(Distribution1D, Unnormalized1)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(2.0f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);
  Normalize(expected);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Unnormalized2)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(1.3f);
  expected.push_back(0.7f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);
  Normalize(expected);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Unnormalized3)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(1.8f);
  expected.push_back(0.0f);
  expected.push_back(3.2f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);
  Normalize(expected);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Unnormalized4)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(1.1f);
  expected.push_back(0.8f);
  expected.push_back(2.3f);
  expected.push_back(0.4f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);
  Normalize(expected);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}

TEST_F(Distribution1D, Unnormalized5)
{
  std::vector<float> expected;
  std::vector<float> foundRatios;
  std::vector<float> foundPdfs;

  expected.push_back(1.0f);
  expected.push_back(1.0f);
  expected.push_back(1.0f);
  expected.push_back(1.0f);
  expected.push_back(1.0f);

  SetValues(expected);
  Sample(foundRatios, foundPdfs);
  Normalize(expected);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundRatios[i], epsilon);
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(expected[i], foundPdfs[i], epsilon);
  }
}