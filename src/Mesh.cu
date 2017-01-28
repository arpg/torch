#include <optix.h>
#include <optix_math.h>
#include <torch/device/Geometry.h>

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtBuffer<float3, 1> vertices;
rtBuffer<float3, 1> normals;
rtBuffer<uint3, 1> faces;

// TORCH_DEVICE bool ReportIntersect(float t)
// {
//   // transform ray to local space
//   const float3 origin = PointToLocal(ray.origin);
//   const float3 direction = NormalToLocal(ray.direction);
//
//   // compute scaled hit point
//   const float3 localPoint = origin + t * direction;
//   const float3 worldPoint = PointToWorld(localPoint);
//
//   // compute time to scaled hit point
//   const float3 delta = worldPoint - ray.origin;
//   const int sign = (dot(delta, ray.direction) < 0) ? -1 : 1;
//   t = sign * sqrt(dot(delta, delta));
//
//   // check if valid intersect
//   if (rtPotentialIntersection(t))
//   {
//     // compute scaled surface normal
//     geometricNormal = NormalToWorld(localPoint);
//     shadingNormal = geometricNormal;
//     rtReportIntersection(0);
//     return true;
//   }
//
//   // invalid intersect
//   return false;
// }

RT_PROGRAM void Intersect(int index)
{
  const uint3& face = faces[index];
  const float3& v0 = vertices[face.x];
  const float3& v1 = vertices[face.y];
  const float3& v2 = vertices[face.z];

  optix::Ray triRay;
  triRay.origin = PointToLocal(ray.origin);
  triRay.direction = normalize(VectorToLocal(ray.direction));
  triRay.tmax = RT_DEFAULT_MAX;
  triRay.tmin = 0;

  float3 n;
  float t;
  float beta;
  float gamma;

  bool hit = intersect_triangle(triRay, v0, v1, v2, n, t, beta, gamma);

  if (hit)
  {
    // compute scaled hit point
    const float3 localPoint = triRay.origin + t * normalize(triRay.direction);
    const float3 worldPoint = PointToWorld(localPoint);

    // compute time to scaled hit point
    const float3 delta = worldPoint - ray.origin;
    const int sign = (dot(delta, ray.direction) < 0) ? -1 : 1;
    t = sign * sqrt(dot(delta, delta));

    // check if valid intersect
    if (rtPotentialIntersection(t))
    {
      // compute scaled surface normal
      geometricNormal = NormalToWorld(n);

      if (normals.size())
      {
        const float alpha = 1 - beta - gamma;
        shadingNormal = alpha * normals[face.x] + beta * normals[face.y] + gamma * normals[face.z];
      }
      else
      {
        shadingNormal = geometricNormal;
      }
      rtReportIntersection(0);
    }
  }
}

RT_PROGRAM void GetBounds(int index, float bounds[6])
{
  const uint3& face = faces[index];
  const float3& v0 = vertices[face.x];
  const float3& v1 = vertices[face.y];
  const float3& v2 = vertices[face.z];

  const float3 bmin = fminf(fminf(v0, v1), v2);
  const float3 bmax = fmaxf(fmaxf(v0, v1), v2);
  BoundsToWorld(bmin, bmax, bounds);
}