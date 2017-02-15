#pragma once

#include <torch/Core.h>

namespace torch
{

class Problem
{
  public:

    Problem(std::shared_ptr<Scene> scene, std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MatteMaterial> material,
        std::shared_ptr<EnvironmentLight> light,
        const std::vector<std::shared_ptr<Keyframe>>& references);

    size_t GetResidualCount() const;

    size_t GetLightParameterCount() const;

    size_t GetAlbedoParameterCount() const;

    void ComputeLightDerivatives();

    void ComputeAlbedoDerivatives();

    optix::Buffer GetRenderBuffer() const;

    optix::Buffer GetLightDerivativeBuffer() const;

    void GetRenderValues(std::vector<float3>& values);

    std::shared_ptr<SparseMatrix> GetAlbedoJacobian(size_t index) const;

    CUdeviceptr GetLightDerivatives();

    CUdeviceptr GetAlbedoDerivatives();

    CUdeviceptr GetKeyframes();

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

    void CreateLightDerivBuffer();

    void CreateKeyframeBuffer();

    void CreateRenderedImageBuffer();

    void CreateBounceImageBuffer();

    void CreateAlbedoBlocks();

    void CreateCameraBuffer();

    void CreatePixelBuffer();

    void CreateRenderBuffer();

    void CreateProgram();

  protected:

    std::shared_ptr<Scene> m_scene;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    std::shared_ptr<EnvironmentLight> m_light;

    std::vector<std::shared_ptr<AlbedoResidualBlock>> m_albedoBlocks;

    std::vector<std::shared_ptr<Camera>> m_cameras;

    std::vector<std::shared_ptr<Keyframe>> m_referenceImages;

    optix::Buffer m_lightDerivs;

    optix::Buffer m_referenceImageBuffer;

    optix::Buffer m_renderedImages;

    optix::Buffer m_bounceImages;

    optix::Buffer m_addToAlbedoBuffer;

    optix::Buffer m_cameraBuffer;

    optix::Buffer m_pixelBuffer;

    optix::Buffer m_renderBuffer;

    optix::Program m_program;

    unsigned int m_programId;

    RTsize m_launchSize;
};

} // namespace torch