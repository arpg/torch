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

    void SetAlbedo(const Spectrum& albedo);

    void SetAlbedo(float r, float g, float b);

    void SetAlbedos(const std::vector<Spectrum>& albedos);

    void SetDerivativeBuffer(optix::Buffer buffer);

    optix::Buffer GetAlbedoBuffer() const;

  private:

    void Initialize();

    void CreateDerivativeBuffer();

    void CreateAlbedoBuffer();

    void UploadAlbedos();

  protected:

    optix::Buffer m_albedoBuffer;

    std::vector<Spectrum> m_albedos;

    optix::Buffer m_derivBuffer;
};

} // namespace torch