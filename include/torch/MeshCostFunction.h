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

    void SetLightSampleCount(unsigned int count);

    void SetMaxNeighborCount(unsigned int count);

    void SetMaxNeighborDistance(float distance);

    void SetSimilarityThreshold(float threshold);

    lynx::Matrix* CreateJacobianMatrix() override;

    void Evaluate(const float * const *parameters, float *residuals);

    void Evaluate(size_t offset, size_t size, float const* const* parameters,
        float* residuals, lynx::Matrix* jacobian) override;

    void Evaluate(float const* const* parameters,
        float* residuals, float* gradient) override;

    void ClearJacobian();

  protected:

    void PrepareEvaluation();

    void ComputeAdjacenies();

    void ComputeLightCoefficients();

    void ResetLightCoefficients();

    void AllocateJacobians();

    void ClearAdjacencies();

    void ClearJacobians();

  private:

    void Initialize();

    void SetDimensions();

    void CreateDummyMaterial();

    void CreateDummyGeometry();

    void CreateShadingBuffer();

    void CreateShadingProgram();

    void ComputeReferenceValues();

    void AllocateSharingValues();

  protected:

    bool m_locked;

    std::shared_ptr<VoxelLight> m_light;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    optix::Buffer m_lightCoeffs;

    optix::Program m_program;

    optix::Program m_closestHitProgram;

    unsigned int m_programId;

    float* m_lightCoeffValues;

    float* m_referenceValues;

    float* m_shadingValues;

    unsigned int m_adjacencyCount;

    unsigned int* m_plist;

    unsigned int* m_qlist;

    float* m_adjacentWeights;

    unsigned int m_iterations;

    unsigned int m_sampleCount;

    unsigned int m_maxNeighborCount;

    float m_maxNeighborDistance;

    float m_similarityThreshold;

    optix::Material m_dummyMaterial;

    optix::Geometry m_dummyGeometry;

    optix::GeometryGroup m_dummyGroup;

    optix::GeometryInstance m_dummyInstance;

    optix::Acceleration m_dummyAccel;

    std::unique_ptr<lynx::Matrix3C> m_maxJacobian;

    std::unique_ptr<lynx::Matrix3C> m_minJacobian;
};

} // namespace torch