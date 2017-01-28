#include <optix.h>
#include <optix_math.h>
#include <torch/device/Geometry.h>

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

TORCH_DEVICE bool ReportIntersect(float t)
{
  // transform ray to local space
  const float3 origin = PointToLocal(ray.origin);
  const float3 direction = normalize(VectorToLocal(ray.direction));

  // compute scaled hit point
  const float3 localPoint = origin + t * direction;
  const float3 worldPoint = PointToWorld(localPoint);

  // compute time to scaled hit point
  const float3 delta = worldPoint - ray.origin;
  const int sign = (dot(delta, ray.direction) < 0) ? -1 : 1;
  t = sign * sqrt(dot(delta, delta));

  // check if valid intersect
  if (rtPotentialIntersection(t))
  {
    // compute scaled surface normal
    geometricNormal = NormalToWorld(localPoint);
    shadingNormal = geometricNormal;
    rtReportIntersection(0);
    return true;
  }

  // invalid intersect
  return false;
}

RT_PROGRAM void Intersect(int index)
{
  // transform ray to local space
  const float3 origin = PointToLocal(ray.origin);
  const float3 direction = normalize(VectorToLocal(ray.direction));

  // compute intermediate intersect results
  const float od = dot(origin, direction);
  const float oo = dot(origin, origin);
  const float a  = od * od - oo + 0.25;

  // check if intersects found
  if (a > 0)
  {
    // compute actual intersects
    const float b  = sqrt(a);
    const float t1 = -od - b;
    const float t2 = -od + b;

    // check if first intersect invalid
    if (!ReportIntersect(t1))
    {
      // report second intersect
      ReportIntersect(t2);
    }
  }
}

RT_PROGRAM void GetBounds(int index, float bounds[6])
{
  const float3 r = make_float3(0.5, 0.5, 0.5);
  BoundsToWorld(-r, r, bounds);
}