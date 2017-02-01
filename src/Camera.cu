#include <optix.h>
#include <torch/device/Core.h>
#include <torch/device/Camera.h>
#include <torch/device/Random.h>
#include <torch/device/Ray.h>

rtDeclareVariable(uint2, pixel, rtLaunchIndex, );
rtDeclareVariable(torch::CameraData, camera, , );
rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(float, sceneEpsilon, , );
rtBuffer<float3, 2> buffer;

class Camera
{
  public:

    TORCH_MEMBER Camera()
    {
      Initialize();
    }

    TORCH_MEMBER float3 Capture()
    {
      m_data.radiance = make_float3(0, 0, 0);

      for (unsigned int i = 0; i < camera.samples; ++i)
      {
        PrepareTrace();
        TracePath();
        FinishTrace();
      }

      return m_data.radiance;
    }

  protected:

    TORCH_MEMBER void PrepareTrace()
    {
      PrepareRay();
      PreparePayload();
      PrepareDirection();
    }

    TORCH_MEMBER void PrepareRay()
    {
      m_ray.ray_type = torch::RAY_TYPE_RADIANCE;
      m_ray.origin = camera.position;
      m_ray.tmin = sceneEpsilon;
      m_ray.tmax = RT_DEFAULT_MAX;
    }

    TORCH_MEMBER void PreparePayload()
    {
      const float t = 1.0f / camera.samples;
      m_data.throughput = make_float3(t, t, t);
      m_data.seed = m_seed;
    }

    TORCH_MEMBER void PrepareDirection()
    {
      const float2 size = make_float2(camera.imageSize);
      const float2 uv = make_float2(pixel) + torch::randf2(m_seed);
      const float2 ratio = (uv / size) - camera.center;
      m_ray.direction = ratio.x * camera.u + ratio.y * camera.v + camera.w;
      m_ray.direction = normalize(m_ray.direction);
    }

    TORCH_MEMBER void TracePath()
    {
      for (unsigned int depth = 0; ; ++depth)
      {
        rtTrace(sceneRoot, m_ray, m_data);
        break;
      }
    }

    TORCH_MEMBER bool ShouldBounce(unsigned int depth)
    {
      if (depth > camera.minDepth)
      {
        const torch::RayBounce& bounce = m_data.bounce;
        const float y = dot(bounce.throughput, bounce.throughput); // TODO
        const float continueProbability = fminf(0.5, y);

        if (torch::randf(m_seed) > continueProbability)
        {
          return false;
        }

        m_data.throughput /= continueProbability;
      }

      if (depth == camera.maxDepth)
      {
        return false;
      }

      return true;
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

    unsigned int m_seed;

    optix::Ray m_ray;

    torch::RadianceData m_data;
};

RT_PROGRAM void Capture()
{
  Camera camera;
  buffer[pixel] = camera.Capture();
}