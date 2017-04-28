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
  payload.occluded = true;

  // if (dot(sample.direction, sample.normal) > 1E-7 &&
  //     dot(sample.direction, sample.snormal) > 1E-7)
  {
    optix::Ray ray;
    ray.origin = sample.origin;
    ray.direction = sample.direction;
    ray.ray_type = RAY_TYPE_SHADOW;
    ray.tmin = sample.tmin;
    ray.tmax = sample.tmax;
    payload.occluded = false;
    rtTrace(sceneRoot, ray, payload);
  }

  return !payload.occluded;
}

} // namespace torch