#pragma once

#include <torch/Core.h>

namespace torch
{

class Problem
{
  public:

    Problem();

    size_t GetResidualCount() const;

    size_t GetLightParameterCount() const;

    size_t GetAlbedoParameterCount() const;

    void ComputeLightDerivatives();

    void ComputeAlbedoDerivatives();

    std::shared_ptr<SparseMatrix> GetAlbedoJacobian(size_t index) const;

    CUdeviceptr GetLightDerivatives();

    CUdeviceptr GetAlbedoDerivatives();

    CUdeviceptr GetReferenceImages();

    CUdeviceptr GetRenderedImages();

    CUdeviceptr GetBounceImages();

  protected:

    void SetLightBufferSize();

    void ZeroLightBufferSize();

    void SetAlbedoBufferSize();

    void ZeroAlbedoBufferSize();

    void SetBounceImageSizes();

    void ZeroBounceImageSizes();

  private:

    void Initialize();

    void CreateScene();

    void CreatePrimitive();

    void CreateLight();

    void CreateCameras();

    void CreateReferenceImages();

    void CreateLightDerivBuffer();

    void CreateAlbedoDerivBuffer();

    void CreateReferenceImageBuffer();

    void CreateRenderedImageBuffer();

    void CreateBounceImageBuffer();

    void CreateAlbedoBlocks();

    void CreateCameraBuffer();

    void CreatePixelBuffer();

    void CreateProgram();

  protected:

    std::shared_ptr<Scene> m_scene;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    std::shared_ptr<EnvironmentLight> m_light;

    std::vector<std::shared_ptr<AlbedoResidualBlock>> m_albedoBlocks;

    std::vector<std::shared_ptr<Camera>> m_cameras;

    std::vector<std::shared_ptr<ReferenceImage>> m_referenceImages;

    optix::Buffer m_lightDerivs;

    optix::Buffer m_albedoDerivs;

    optix::Buffer m_referenceImageBuffer;

    optix::Buffer m_renderedImages;

    optix::Buffer m_bounceImages;

    optix::Buffer m_addToAlbedoBuffer;

    optix::Buffer m_cameraBuffer;

    optix::Buffer m_pixelBuffer;

    optix::Program m_program;

    unsigned int m_programId;

    RTsize m_launchSize;
};

} // namespace torch