#include <torch/device/ReflectanceCostFunction.cuh>
#include <optix_math.h>
#include <lynx/Exception.h>

namespace torch
{
  // launch:   3 * number of vertices
  // params:   3 * number of vertices
  // residual: 3 * number of vertices
  // jacobian: non-zero value buffer (3 * number of map-indices)
  // size:     3 * number of vertices
  // map:      column indices (row "owner" indicated by offsets)
  // offsets:  number of vertices + 1 (points to addresses in map)
  // chromThr: minimum acceptable dot(ch1, ch2) that defines "similar" albedos
  // weight:   scalar penalty apply to difference between albedos

__global__ void EvaluateKernel(const float* params, float* residuals,
    float* jacobian, size_t size, const uint* map, const uint* offsets,
    float chromThreshold, float weight)
{
  // vertex-channel index in CONSTRAINTS
  const unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;



  // check if thread is valid
  if (threadIndex < size)
  {


    // vertex index in CONSTRAINTS
    const unsigned int vertexIndex = threadIndex / 3;

    // channel offset for vertex index above
    const unsigned int channelIndex = threadIndex % 3;

    // vertex-channel index of R channel in RGB albedo params
    const unsigned int vertexFirstChannelIndex = 3 * vertexIndex;

    // pointer to RGB albedo for vertex
    const float3 albedo =
        reinterpret_cast<const float3&>(params[vertexFirstChannelIndex]);

    // norm of albedo for checking blackness
    const float albedoNorm = length(albedo);

    // indicates if albedo is black
    const bool isBlack = albedoNorm < 0.1f;

    // chromaticity of vertex
    float3 chrom = (isBlack) ? make_float3(0, 0, 0) : albedo / albedoNorm;

    // first index in map associated with current vertex
    const unsigned int mapStart = offsets[vertexIndex];

    // firrst index in map NOT associated with current vertex after mapStart
    const unsigned int mapStop = offsets[vertexIndex + 1];

    // declare self map index for future assignment
    unsigned int selfMapIndex = 100;

    // initialize number of brighter adjacent vertices to zero
    unsigned int brighterAdjacentCount = 0;


    // compare vertex against all adjacent vertices
    for (unsigned int mapIndex = mapStart; mapIndex < mapStop; ++mapIndex)
    {
      // check if jacobian requested
      if (jacobian)
      {


        // get index of jacobian value
        const unsigned int jacobianIndex = 3 * mapIndex + channelIndex;

        // compute and store jacobian
        jacobian[jacobianIndex] = 0.0f;


      }


      // vertex index of adjacent vertex in PARAMETERS
      const unsigned int adjVertexIndex = map[mapIndex];

      // check if adjacent vertex is self
      if (adjVertexIndex == vertexIndex)
      {

        // assign map index of self
        selfMapIndex = mapIndex;

        // skip remainder of loop
        continue;


      }

      // adjacent vertex-channel index of R channel in RGB albedo params
      const unsigned int adjVertexFirstChannelIndex = 3 * adjVertexIndex;

      // pointer to RGB albedo for adjacent vertex
      const float3 adjAlbedo =
          reinterpret_cast<const float3&>(params[adjVertexFirstChannelIndex]);

      // norm of adjacent vertex albedo for checking blackness
      const float adjAlbedoNorm = length(adjAlbedo);

      // indicates if adjacent albedo is black
      const bool adjIsBlack = adjAlbedoNorm < 0.1f;


      // check if adjacent vertex is not black
      if (!adjIsBlack)
      {


        // chromaticity of adjacent vertex
        // use identical chromaticity if adjacent vertex is black
        const float3 adjChrom = adjAlbedo / adjAlbedoNorm;

        // copy adjacent chromaticity if vertex is black
        chrom = (isBlack) ? adjChrom : albedo / albedoNorm;

        // get dot product of both chromaticities
        const float dotChrom = dot(chrom, adjChrom);



        // check if chromaticities are similiar enough
        if (dotChrom > chromThreshold)
        {


          // color channel value of vertex-channel
          const float channel =
              reinterpret_cast<const float*>(&albedo)[channelIndex];

          // color channel value of adjacent vertex-channel
          const float adjChannel =
              reinterpret_cast<const float*>(&adjAlbedo)[channelIndex];



          // check if vertex is darker than adjacent vertex
          if (channel < adjChannel)
          {


            // get index of residual for vertex-channel
            const unsigned int residualIndex = threadIndex;

            // compute and store residual
            residuals[residualIndex] += weight * (adjChannel - channel);

            // increment brighter adjacent vertex count
            ++brighterAdjacentCount;



            // check if jacobian requested
            if (jacobian)
            {


              // get index of jacobian value
              const unsigned int jacobianIndex = 3 * mapIndex + channelIndex;

              // compute and store jacobian
              jacobian[jacobianIndex] = weight;


            }
          }
        }
      }
    }

    // check if jacobian requested
    if (jacobian)
    {


      // get index of jacobian value
      const unsigned int jacobianIndex = 3 * selfMapIndex + channelIndex;

      // compute and store jacobian
      jacobian[jacobianIndex] = -1.0f * brighterAdjacentCount;


    }
  }
}

void Evaluate(const float* params, float* residual, float* jacobian,
    size_t size, const uint* map, const uint* offsets, float chromThreshold,
    float weight)
{
  LYNX_ASSERT(size <= (1024 * 65535), "unsupported launch size");
  const size_t blockDim = (size > 1024) ? 1024 : size;
  const size_t gridDim = (size + blockDim - 1) / blockDim;

  EvaluateKernel<<<blockDim, gridDim>>>(params, residual, jacobian, size,
      map, offsets, chromThreshold, weight);
}

} // namespace torch