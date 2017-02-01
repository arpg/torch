#include <optix.h>
#include <torch/device/Core.h>
#include <torch/device/Camera.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

rtDeclareVariable(uint2, pixel, rtLaunchIndex, );
rtDeclareVariable(uint2, imageSize, rtLaunchDim, );

rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(unsigned int, sampleCount, , );
rtDeclareVariable(torch::CameraData, camera, , );
rtBuffer<float3, 2> buffer;

class Camera
{
  public:

    TORCH_MEMBER Camera() :
      m_sampleCount(1),
      m_sceneEpsilon(1E-4)
    {
      Initialize();
    }

    TORCH_MEMBER void SetSceneEpsilon(float epsilon)
    {
      m_sceneEpsilon = epsilon;
    }

    TORCH_MEMBER void SetSampleCount(unsigned int count)
    {
      m_sampleCount = count;
    }

    TORCH_MEMBER float3 Capture(const uint2& pixel)
    {
      m_data.radiance = make_float3(0, 0, 0);

      for (unsigned int i = 0; i < m_sampleCount; ++i)
      {
        TraceRay();
      }

      return m_data.radiance;
    }

  protected:

    TORCH_MEMBER void TraceRay()
    {
      PrepareTrace();
      rtTrace(sceneRoot, m_ray, m_data);
      FinishTrace();
    }

    TORCH_MEMBER void PrepareTrace()
    {
      PrepareRay();
      PreparePayload();
    }

    TORCH_MEMBER void PrepareRay()
    {
      m_ray.ray_type = torch::RAY_TYPE_RADIANCE;
      m_ray.origin = camera.position;
      m_ray.tmin = m_sceneEpsilon;
      m_ray.tmax = RT_DEFAULT_MAX;
      UpdateDirection();
    }

    TORCH_MEMBER void PreparePayload()
    {
      const float t = 1.0f / m_sampleCount;
      m_data.throughput = make_float3(t, t, t);
      m_data.seed = m_seed;
    }

    TORCH_MEMBER void UpdateDirection()
    {
      const float2 size = make_float2(imageSize);
      const float2 uv = make_float2(pixel) + torch::randf2(m_seed);
      const float2 ratio = (uv / size) - camera.center;
      m_ray.direction = ratio.x * camera.u + ratio.y * camera.v + camera.w;
      m_ray.direction = normalize(m_ray.direction);
    }

    TORCH_MEMBER void FinishTrace()
    {
      m_seed = m_data.seed;
    }

  private:

    TORCH_MEMBER void Initialize()
    {
      InitializeSeed();
    }

    TORCH_MEMBER void InitializeSeed()
    {
      const unsigned int a = pixel.x;
      const unsigned int b = pixel.y;
      m_seed = torch::init_seed<16>(a, b);
    }

  protected:

    optix::Ray m_ray;

    torch::RadianceData m_data;

    unsigned int m_seed;

    unsigned int m_sampleCount;

    float m_sceneEpsilon;
};

RT_PROGRAM void Capture()
{
  Camera myCamera;
  myCamera.SetSceneEpsilon(sceneEpsilon);
  myCamera.SetSampleCount(sampleCount);
  buffer[pixel] = myCamera.Capture(pixel);
}