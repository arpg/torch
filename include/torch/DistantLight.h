#pragma once

#include <string>
#include <torch/Light.h>

namespace torch
{

class DistantLight : public Light
{
  public:

    DistantLight(std::shared_ptr<Context> context);

    Spectrum GetPower() const override;

    Spectrum GetRadiance() const;

    void SetRadiance(const Spectrum& intensity);

    void SetRadiance(float r, float g, float b);

    float3 GetDirection() const;

    void SetDirection(const float3& direction);

    void SetDirection(float x, float y, float z);

    void BuildScene(Link& link) override;

  protected:

    Spectrum m_radiance;
};

} // namespace torch