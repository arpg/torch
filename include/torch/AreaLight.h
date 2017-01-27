#pragma once

#include <torch/Light.h>

namespace torch
{

class Geometry;

class AreaLight : public Light
{
  public:

    AreaLight(std::shared_ptr<Context> context);

    Spectrum GetPower() const override;

    Spectrum GetRadiance() const;

    void SetRadiance(const Spectrum& radiance);

    void SetRadiance(float r, float g, float b);

    std::shared_ptr<Geometry> GetGeometry() const;

    void SetGeometry(std::shared_ptr<Geometry> geometry);

    void BuildScene(Link& link) override;

  protected:

    Spectrum m_radiance;

    std::shared_ptr<Geometry> m_geometry;
};

} // namespace torch