#pragma once

#include <torch/Light.h>
#include <torch/Spectrum.h>

namespace torch
{

class EnvironmentLight : public Light
{
  public:

    EnvironmentLight(std::shared_ptr<Context> context);

    unsigned int GetRowCount() const;

    void SetRowCount(unsigned int count);

    unsigned int GetDirectionCount() const;

    void SetRadiance(const Spectrum& radiance);

    void SetRadiance(float r, float g, float b);

    void SetRadiance(size_t index, const Spectrum& radiance);

    void SetRadiance(size_t index, float r, float g, float b);

    void SetRadiance(const std::vector<Spectrum>& radiance);

    void BuildScene(Link& link) override;

  private:

    void Initialize();

    void UpdateDirectionCount();

  protected:

    unsigned int m_rowCount;

    unsigned int m_directionCount;

    std::vector<Spectrum> m_radiance;

    static const unsigned int minRowCount;
};

} // namespace torch