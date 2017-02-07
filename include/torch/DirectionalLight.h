#pragma once

#include <torch/Light.h>
#include <torch/Spectrum.h>
#include <torch/Vector.h>

namespace torch
{

class DirectionalLight : public Light
{
  public:

    DirectionalLight(std::shared_ptr<Context> context);

    Spectrum GetRadiance() const;

    void SetRadiance(const Spectrum& intensity);

    void SetRadiance(float r, float g, float b);

    Vector GetDirection() const;

    void SetDirection(const Vector& direction);

    void SetDirection(float x, float y, float z);

    void BuildScene(Link& link) override;

  protected:

    Spectrum m_radiance;
};

} // namespace torch