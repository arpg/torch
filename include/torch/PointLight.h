#pragma once

#include <string>
#include <torch/Light.h>

namespace torch
{

class PointLight : public Light
{
  public:

    PointLight(std::shared_ptr<Context> context);

    Spectrum GetPower() const override;

    Spectrum GetIntensity() const;

    void SetIntensity(const Spectrum& intensity);

    void SetIntensity(float r, float g, float b);

    void BuildScene(Link& link) override;

  protected:

    Spectrum m_intensity;
};

} // namespace torch