#pragma once

#include <torch/Material.h>
#include <torch/Spectrum.h>

namespace torch
{

class MatteMaterial : public Material
{
  public:

    MatteMaterial(std::shared_ptr<Context> context);

    size_t GetAlbedoCount() const;

    void GetAlbedos(std::vector<Spectrum>& albedos) const;

    void SetAlbedo(const Spectrum& albedo);

    void SetAlbedo(float r, float g, float b);

    void SetAlbedos(const std::vector<Spectrum>& albedos);

    void SetDerivativeProgram(optix::Program program);

    void SetCameraBuffer(optix::Buffer buffer);

    void SetPixelBuffer(optix::Buffer buffer);

    optix::Buffer GetAlbedoBuffer() const;

    void LoadAlbedos();

  private:

    void Initialize();

    void CreateCameraBuffer();

    void CreatePixelBuffer();

    void CreateAlbedoBuffer();

    void UploadAlbedos();

  protected:

    optix::Buffer m_albedoBuffer;

    std::vector<Spectrum> m_albedos;

    optix::Buffer m_derivBuffer;

    optix::Buffer m_cameraBuffer;

    optix::Buffer m_pixelBuffer;
};

} // namespace torch