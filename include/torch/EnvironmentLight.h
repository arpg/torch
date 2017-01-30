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

    unsigned int GetDirectionCount(unsigned int row) const;

    void SetRadiance(const Spectrum& radiance);

    void SetRadiance(float r, float g, float b);

    void SetRadiance(size_t index, const Spectrum& radiance);

    void SetRadiance(const std::vector<Spectrum>& radiance);

    void BuildScene(Link& link) override;

  private:

    void Initialize();

    void UpdateOffsets();

  protected:

    unsigned int m_rowCount;

    std::vector<Spectrum> m_radiance;

    std::vector<unsigned int> m_offsets;
};

} // namespace torch