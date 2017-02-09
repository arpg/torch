#include <torch/device/Core.h>
#include <torch/device/Light.h>
#include <torch/device/Ray.h>

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

namespace torch
{

TORCH_DEVICE bool IsVisible(const LightSample& sample)
{
  ShadowData payload;
  payload.occluded = false;

  if (dot(sample.direction, sample.normal) > 0)
  {
    optix::Ray ray;
    ray.origin = sample.origin;
    ray.direction = sample.direction;
    ray.ray_type = RAY_TYPE_SHADOW;
    ray.tmin = sample.tmin;
    ray.tmax = sample.tmax;
    rtTrace(sceneRoot, ray, payload);
  }

  return !payload.occluded;
}

} // namespace torch