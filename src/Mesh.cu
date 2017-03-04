#include <optix.h>
#include <optix_math.h>
#include <torch/device/Ray.h>
#include <torch/device/Transform.h>

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, );
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, );
rtDeclareVariable(float2, triScales, attribute triScales, );
rtDeclareVariable(uint3, triFace, attribute triFace, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(torch::DepthData, depthData, rtPayload, );
rtBuffer<float3, 1> vertices;
rtBuffer<float3, 1> normals;
rtBuffer<uint3, 1> faces;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

RT_PROGRAM void Intersect(int index)
{
  const uint3& face = faces[index];
  const float3& v0 = vertices[face.x];
  const float3& v1 = vertices[face.y];
  const float3& v2 = vertices[face.z];

  optix::Ray triRay;
  triRay.origin = PointToLocal(ray.origin);
  triRay.direction = normalize(VectorToLocal(ray.direction));
  triRay.direction = normalize(VectorToLocal(ray.direction));
  triRay.tmax = RT_DEFAULT_MAX;
  triRay.tmin = 0.0f;

  float3 n;
  float t;
  float beta;
  float gamma;

  // bool hit = IntersectTriangle(triRay, v0, v1, v2, n, t, beta, gamma);
  bool hit = intersect_triangle(triRay, v0, v1, v2, n, t, beta, gamma);
  n = normalize(n);
  if (dot(n, -ray.direction) < 0.0f) n = -n;

  if (hit)
  {
    // compute scaled hit point
    const float3 localPoint = triRay.origin + t * normalize(triRay.direction);
    const float3 worldPoint = PointToWorld(localPoint);

    // compute time to scaled hit point
    const float3 delta = worldPoint - ray.origin;
    const int sign = (dot(delta, ray.direction) < 0) ? -1 : 1;
    t = sign * sqrtf(dot(delta, delta));

    // check if valid intersect
    if (rtPotentialIntersection(t))
    {
      // compute scaled surface normal
      geometricNormal = NormalToWorld(n);
      triScales = make_float2(beta, gamma);
      triFace = face;

      if (normals.size())
      {
        const float alpha = 1.0f - beta - gamma;
        shadingNormal = alpha * normals[face.x] + beta * normals[face.y] + gamma * normals[face.z];
        shadingNormal = faceforward(shadingNormal, -ray.direction, geometricNormal);
        shadingNormal = normalize(shadingNormal);

        if (isnan(shadingNormal.x) ||
            isnan(shadingNormal.y) ||
            isnan(shadingNormal.z))
        {
          shadingNormal = geometricNormal;
        }
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
