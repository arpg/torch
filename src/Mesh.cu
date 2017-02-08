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

TORCH_DEVICE bool IntersectTriangle(const optix::Ray& ray,
  const float3& p0, const float3& p1, const float3& p2, float3& n, float& t,
  float& beta, float& gamma)
{
  const float3 e0 = p1 - p0;
  const float3 e1 = p0 - p2;
  n  = cross( e1, e0 );

  const float3 e2 = ( 1.0f / dot( n, ray.direction ) ) * ( p0 - ray.origin );
  const float3 i  = cross( ray.direction, e2 );

  beta  = dot( i, e1 );
  gamma = dot( i, e0 );
  t     = dot( n, e2 );

  if (beta > -1E-4) beta = 0.0f;
  if (gamma > -1E-4) gamma = 0.0f;
  const float alpha = 1 - (beta + gamma);

  if (alpha < 0 && alpha > -1E-4)
  {
    if (beta > gamma) beta -= alpha;
    else gamma -= alpha;
  }

  return ( (t<ray.tmax) & (t>ray.tmin) & (beta>=0.0f) & (gamma>=0.0f) & (beta+gamma<=1) );
}

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

  bool hit = IntersectTriangle(triRay, v0, v1, v2, n, t, beta, gamma);
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
