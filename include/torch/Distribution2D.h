#pragma once

#include <torch/Core.h>

namespace torch
{

class Distribution2D
{
  public:

    Distribution2D(std::shared_ptr<Context> context);

    optix::Program GetProgram() const;

    void SetValues(const std::vector<float>& values,
        const std::vector<unsigned int>& offsets);

  private:

    void Initialize();

    void CreateProgram();

    void CreateRowBuffer();

    void CreateColumnBuffer();

    void CreateOffsetBuffer();

    template <typename T>
    static void CopyVector(const std::vector<T>& values, optix::Buffer buffer);

  protected:

    std::shared_ptr<Context> m_context;

    optix::Program m_program;

    optix::Buffer m_rowCdf;

    optix::Buffer m_colCdfs;

    optix::Buffer m_offsets;
};

} // namespace torch