#include <torch/SceneGeometrySampler.h>
#include <torch/Context.h>
#include <torch/Distribution1D.h>
#include <torch/GeometryGroupSampler.h>
#include <torch/MeshSampler.h>
#include <torch/PtxUtil.h>
#include <torch/SphereSampler.h>
#include <torch/device/Geometry.h>

namespace torch
{

SceneGeometrySampler::SceneGeometrySampler(std::shared_ptr<Context> context) :
  m_context(context)
{
  Initialize();
}

optix::Program SceneGeometrySampler::GetProgram() const
{
  return m_program;
}

void SceneGeometrySampler::Add(const GeometryGroupData& group)
{
  GeometrySampler* sampler = m_samplers[GEOM_TYPE_GROUP].get();
  static_cast<GeometryGroupSampler*>(sampler)->Add(group);
}

void SceneGeometrySampler::Add(const MeshData& mesh)
{
  GeometrySampler* sampler = m_samplers[GEOM_TYPE_MESH].get();
  static_cast<MeshSampler*>(sampler)->Add(mesh);
}

void SceneGeometrySampler::Add(const SphereData& sphere)
{
  GeometrySampler* sampler = m_samplers[GEOM_TYPE_SPHERE].get();
  static_cast<SphereSampler*>(sampler)->Add(sphere);
}

void SceneGeometrySampler::Clear()
{
  for (std::unique_ptr<GeometrySampler>& sampler : m_samplers)
  {
    sampler->Clear();
  }
}

void SceneGeometrySampler::Update()
{
  for (std::unique_ptr<GeometrySampler>& sampler : m_samplers)
  {
    sampler->Update();
  }
}

void SceneGeometrySampler::Initialize()
{
  CreateProgram();
  CreateSamplers();
}

void SceneGeometrySampler::CreateProgram()
{
  const std::string file = PtxUtil::GetFile("SceneGeometrySampler");
  m_program = m_context->CreateProgram(file, "Sample");
}

void SceneGeometrySampler::CreateSamplers()
{
  m_samplers.resize(GEOM_TYPE_COUNT);
  std::unique_ptr<GeometrySampler> sampler;

  sampler = std::make_unique<GeometryGroupSampler>(m_context);
  m_program["SampleGeometryGroups"]->set(sampler.get()->GetProgram());
  m_samplers[GEOM_TYPE_GROUP] = std::move(sampler);

  sampler = std::make_unique<MeshSampler>(m_context);
  m_program["SampleMeshes"]->set(sampler.get()->GetProgram());
  m_samplers[GEOM_TYPE_MESH] = std::move(sampler);

  sampler = std::make_unique<SphereSampler>(m_context);
  m_program["SampleSpheres"]->set(sampler.get()->GetProgram());
  m_samplers[GEOM_TYPE_SPHERE] = std::move(sampler);
}

} // namespace torch