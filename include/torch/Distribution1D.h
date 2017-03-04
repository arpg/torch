#pragma once

#include <torch/Core.h>

namespace torch
{

class Distribution1D
{
  public:

    Distribution1D(std::shared_ptr<Context> context, bool useSecondName = false);

    optix::Program GetProgram() const;

    void SetValues(const std::vector<float>& values);

  protected:

    void Upload(const std::vector<float>& cdf);

    static void Normalize(std::vector<float>& cdf);

  private:

    void Initialize();

    void CreateProgram();

    void CreateBuffer();

  protected:

    bool m_useSecondName;

    std::shared_ptr<Context> m_context;

    optix::Buffer m_buffer;

    optix::Program m_program;
};

} // namespace torch