#pragma once

#include <torch/Core.h>

namespace torch
{

class Distribution
{
  public:

    Distribution(std::shared_ptr<Context> context);

    optix::Program GetProgram() const;

    void SetValues(const std::vector<float>& values);

  protected:

    void Upload(const std::vector<float>& cdf);

    static void Normalize(std::vector<float>& cdf);

  private:

    void Initialize();

    void CreateBuffer();

    void CreateProgram();

  protected:

    std::shared_ptr<Context> m_context;

    optix::Buffer m_buffer;

    optix::Program m_program;
};

} // namespace torch