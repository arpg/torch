#include <gtest/gtest.h>
#include <torch/Torch.h>
#include <TestPtxUtil.h>

class Distribution1D : public testing::Test
{
  public:

    Distribution1D() :
      m_sampleCount(1),
      m_bufferSize(0)
    {
      Initialize();
    }

    void SetValues(const std::vector<float>& values)
    {
      m_bufferSize = values.size();
      m_distribution->SetValues(values);
      m_countBuffer->setSize(m_bufferSize);
      m_pdfBuffer->setSize(m_bufferSize);
      SetSampleCount();
    }

    void Sample(std::vector<float>& ratios, std::vector<float>& pdfs)
    {
      ClearBuffers();
      LaunchProgram();
      GetSampleRatios(ratios);
      GetSamplePdfs(pdfs);
    }

    static void Normalize(std::vector<float>& values)
    {
      float sum = 0;
      for (float value : values) sum += value;
      for (size_t i = 0; i < values.size(); ++i) values[i] /= sum;
    }

  protected:

    void SetSampleCount()
    {
      m_sampleCount = 10000 * m_bufferSize / epsilon;
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
      ratios.resize(m_bufferSize);
      std::vector<unsigned int> counts(m_bufferSize);
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

    template <typename T>
    void FillBuffer(optix::Buffer buffer, T value)
    {
      T* device = reinterpret_cast<T*>(buffer->map());
      std::fill(device, device + m_bufferSize, value);
      buffer->unmap();
    }

    template <typename T>
    void CopyBuffer(optix::Buffer buffer, std::vector<T>& counts)
    {
      counts.resize(m_bufferSize);
      T* device = reinterpret_cast<T*>(buffer->map());
      std::copy(device, device + m_bufferSize, counts.data());
      buffer->unmap();
    }

  private:

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
      m_countBuffer->setSize(m_bufferSize);
      m_program["counts"]->setBuffer(m_countBuffer);
    }

    void CreatePdfBuffer()
    {
      m_pdfBuffer = m_context->CreateBuffer(RT_BUFFER_INPUT_OUTPUT);
      m_pdfBuffer->setFormat(RT_FORMAT_FLOAT);
      m_pdfBuffer->setSize(m_bufferSize);
      m_program["pdfs"]->setBuffer(m_pdfBuffer);
    }

    void CreateDistribution()
    {
      m_distribution = std::make_unique<torch::Distribution1D>(m_context);
      m_program["Sample1D"]->set(m_distribution->GetProgram());
    }

  protected:

    unsigned int m_sampleCount;

    unsigned int m_programId;

    size_t m_bufferSize;

    optix::Program m_program;

    optix::Buffer m_countBuffer;

    optix::Buffer m_pdfBuffer;

    std::shared_ptr<torch::Context> m_context;

    std::unique_ptr<torch::Distribution1D> m_distribution;

    static constexpr float epsilon = 1E-4;
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