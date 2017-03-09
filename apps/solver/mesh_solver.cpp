#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/Torch.h>

DEFINE_string(mesh, "mesh.ply", "input mesh");
DEFINE_string(hd_mesh, "", "high-res mesh used for computing final results");
DEFINE_int32(voxel_dim, 7, "number of voxels along each grid dimension");
DEFINE_double(voxel_size, 0.5, "size of each individual voxel");
DEFINE_double(max_dist, 0.1, "max distance for neighbor evaluation");
DEFINE_double(sim_thresh, 0.1, "minimum similarity value for neighbor status");
DEFINE_int32(knn, 5, "number of neighboring vertices to contrain");
DEFINE_int32(samples, 512, "number of light samples per vertex");
DEFINE_bool(use_act, false, "use activation cost regulizer");
DEFINE_double(inner_act, 1.0, "inner log scale for activation cost");
DEFINE_double(outer_act, 1.0, "outer log scale for activation cost");
DEFINE_double(min_light_change, 1E-5, "minimum change rate for light params");
DEFINE_bool(verbose, false, "verbose output of optimization progress");
DEFINE_int32(max_iters, 1, "max number of iterations");
DEFINE_string(out_mesh, "out_mesh.ply", "file to write final mesh");

using namespace torch;

int iteration = 0;

std::shared_ptr<Scene> scene;
std::shared_ptr<VoxelLight> light;
std::shared_ptr<MatteMaterial> material;
std::shared_ptr<Mesh> mesh;

lynx::Problem* problem;
MeshCostFunction* costFunction;
float* lightValues;

void CreateScene()
{
  LOG(INFO) << "Building scene...";

  scene = std::make_shared<Scene>();

  mesh = scene->CreateMesh(FLAGS_mesh);
  std::shared_ptr<Material> temp = scene->CreateMaterial(FLAGS_mesh);
  material = std::static_pointer_cast<MatteMaterial>(temp);

  std::shared_ptr<Primitive> primitive;
  primitive = scene->CreatePrimitive();
  primitive->SetGeometry(mesh);
  primitive->SetMaterial(material);
  scene->Add(primitive);

  light = scene->CreateVoxelLight();
  light->SetDimensions(FLAGS_voxel_dim);
  light->SetVoxelSize(FLAGS_voxel_size);
  light->SetRadiance(1, 1, 1);
  scene->Add(light);

  light->GetContext()->Compile();
}

void CreateProblem()
{
  LOG(INFO) << "Creating light problem...";

  const size_t paramCount = 3 * light->GetVoxelCount();
  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  lightValues = reinterpret_cast<float*>(pointer);

  problem = new lynx::Problem();
  problem->AddParameterBlock(lightValues, paramCount);
  problem->SetLowerBound(lightValues, 0.0f);

  costFunction = new torch::MeshCostFunction(light, mesh, material);
  costFunction->SetMaxNeighborCount(FLAGS_knn);
  costFunction->SetMaxNeighborDistance(FLAGS_max_dist);
  costFunction->SetSimilarityThreshold(FLAGS_sim_thresh);
  costFunction->SetLightSampleCount(FLAGS_samples);
  problem->AddResidualBlock(costFunction, nullptr, lightValues);

  if (FLAGS_use_act)
  {
    VoxelActivationCostFunction* actCostFunction;
    actCostFunction = new VoxelActivationCostFunction(light);
    actCostFunction->SetBias(1.0);
    actCostFunction->SetInnerScale(FLAGS_inner_act);
    actCostFunction->SetOuterScale(FLAGS_outer_act);
    problem->AddResidualBlock(actCostFunction, nullptr, lightValues);
  }
}

void SolveProblemOnce()
{
  LOG(INFO) << "Solving light problem...";

  costFunction->ClearJacobian();

  lynx::Solver::Options options;
  options.maxIterations = 50000;
  options.minCostChangeRate = FLAGS_min_light_change;
  options.verbose = FLAGS_verbose;

  lynx::Solver solver(problem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  LOG(INFO) << summary.BriefReport();

  const size_t count = problem->GetParameterBlockSize(0);
  std::vector<Spectrum> radiance(count / 3);

  const size_t bytes = sizeof(float) * count;
  const cudaMemcpyKind type = cudaMemcpyDeviceToHost;
  LYNX_CHECK_CUDA(cudaMemcpy(radiance.data(), lightValues, bytes, type));
  light->SetRadiance(radiance);
}

void SaveResults()
{
  LOG(INFO) << "Computing final albedos...";

  std::shared_ptr<Mesh> final_mesh;
  std::shared_ptr<MatteMaterial> final_material;

  if (FLAGS_hd_mesh.empty())
  {
    final_mesh = mesh;
    final_material = material;
  }
  else
  {
    final_mesh = scene->CreateMesh(FLAGS_hd_mesh);
    std::shared_ptr<Material> temp = scene->CreateMaterial(FLAGS_hd_mesh);
    final_material = std::static_pointer_cast<MatteMaterial>(temp);
  }

  ShadingRemover remover(final_mesh, final_material);
  remover.SetSampleCount(FLAGS_samples);
  remover.Remove();
  final_material->LoadAlbedos();

  LOG(INFO) << "Writing final mesh estimate...";

  MeshWriter writer(final_mesh, final_material);
  writer.Write(FLAGS_out_mesh);

  LOG(INFO) << "Writing final voxel values...";

  std::vector<float> values(3 * light->GetVoxelCount());
  optix::Buffer radiance = light->GetRadianceBuffer();
  float* device = reinterpret_cast<float*>(radiance->map());
  std::copy(device, device + values.size(), values.data());
  radiance->unmap();

  for (size_t i = 0; i < light->GetVoxelCount(); ++i)
  {
    std::cout << values[3 * i + 0] << " ";
    std::cout << values[3 * i + 1] << " ";
    std::cout << values[3 * i + 2] << std::endl;
  }
}

void SolveProblem()
{
  CreateScene();
  CreateProblem();

  for (iteration = 0; iteration < FLAGS_max_iters; ++iteration)
  {
    LOG(INFO) << "Starting iteration " << iteration << "...";
    SolveProblemOnce();
  }

  SaveResults();
}

int main(int argc, char** argv)
{
  LOG(INFO) << "Starting...";
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SolveProblem();

  LOG(INFO) << "Success";
  return 0;
}