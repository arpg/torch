#pragma once

#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class ReflectanceCostFunction : public lynx::CostFunction
{
  public:

    ReflectanceCostFunction(std::shared_ptr<MatteMaterial> material,
        std::shared_ptr<Mesh> mesh);

    virtual ~ReflectanceCostFunction();

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(float const* const* parameters, float* residuals) override;

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

  private:

    void Initialize();

    void SetDimensions();

    void CreateAdjacencyMap();

  protected:

    std::shared_ptr<MatteMaterial> m_material;

    std::shared_ptr<Mesh> m_mesh;

    std::vector<std::vector<size_t>> m_adjacencyMap;
};

} // namespace torch