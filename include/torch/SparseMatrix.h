#pragma once

#include <torch/Core.h>

namespace torch
{

class SparseMatrix
{
  public:

    SparseMatrix(std::shared_ptr<Context> context);

    void Allocate(const std::vector<unsigned int>& rowOffsets,
        const std::vector<unsigned int>& colIndices);

    optix::Program GetAddProgram() const;

    void GetValues(std::vector<float3>& values);

    void GetRowOffsets(std::vector<unsigned int>& offsets);

    void GetColumnIndices(std::vector<unsigned int>& indices);

  protected:

    void AllocateValues(size_t size);

    void SetRowOffsets(const std::vector<unsigned int>& offsets);

    void SetColumnIndices(const std::vector<unsigned int>& indices);

  private:

    void Initialize();

    void CreateValueBuffer();

    void CreateRowOffsetBuffer();

    void CreateColumnIndexBuffer();

    void CreateProgram();

  protected:

    std::shared_ptr<Context> m_context;

    optix::Program m_program;

    optix::Buffer m_values;

    optix::Buffer m_rowOffsets;

    optix::Buffer m_colIndices;
};

} // namespace torch