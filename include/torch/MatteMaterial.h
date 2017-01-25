#pragma once

#include <torch/Material.h>
#include <torch/Spectrum.h>

namespace torch
{

class MatteMaterial : public Material
{
  public:

    MatteMaterial(std::shared_ptr<Context> context);

    ~MatteMaterial();

    Spectrum GetAlbedo() const;

    void SetAlbedo(const Spectrum& albedo);

    void SetAlbedo(float r, float g, float b);

  private:

    void Initialize();

  protected:

    Spectrum m_albedo;
};

} // namespace torch