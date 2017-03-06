#pragma once

#include <lynx/lynx.h>
#include <torch/Core.h>

namespace torch
{

class MeshCostFunction : public lynx::CostFunction
{
  public:

    MeshCostFunction(std::shared_ptr<VoxelLight> light,
        std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MatteMaterial> material);

    virtual ~MeshCostFunction();

    void SetMaxNeighborCount(unsigned int count);

    void SetMaxNeighborDistance(float distance);

    void SetSimilarityThreshold(float threshold);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(const float * const *parameters, float *residuals);

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

    void ClearJacobian();

  protected:

    void PrepareEvaluation();

    void ComputeAdjacenies();

    void ComputeLightCoefficients();

    void ResetLightCoefficients();

    void ClearAdjacencies();

  private:

    void Initialize();

    void SetDimensions();

    void CreateBuffer();

    void CreateProgram();

    void ComputeReferenceValues();

  protected:

    bool m_locked;

    std::shared_ptr<VoxelLight> m_light;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    optix::Buffer m_lightCoeffs;

    optix::Program m_program;

    unsigned int m_programId;

    float* m_lightCoeffValues;

    float* m_referenceValues;

    unsigned int m_adjacencyCount;

    unsigned int* m_adjacentVertices;

    float* m_adjacentWeights;

    unsigned int m_iterations;

    unsigned int m_maxNeighborCount;

    float m_maxNeighborDistance;

    float m_similarityThreshold;
};

} // namespace torch