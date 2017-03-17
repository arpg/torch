#include <ctime>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lynx/lynx.h>
#include <sophus/se3.hpp>
#include <torch/Torch.h>

DEFINE_string(mesh, "mesh.ply", "input mesh");
DEFINE_string(poses, "poses.csv", "input pose file");
DEFINE_int32(voxel_dim, 3, "number of voxels along each grid dimension");
DEFINE_double(voxel_size, 0.1, "size of each individual voxel");
DEFINE_bool(meshlab, false, "mesh files should to be writen for meshlab");
DEFINE_double(fx, 535.7239, "camera horizontal focal length");
DEFINE_double(fy, 536.2900, "camera vertical focal length");
DEFINE_double(cx, 320.2685, "camera horizontal center point");
DEFINE_double(cy, 240.2924, "camera vertical center point");
DEFINE_double(img_scale, 1, "downscale ration of reference cameras");
DEFINE_int32(samples, 4, "square-root of number of samples per pixel");
DEFINE_bool(euler, false, "interpret pose file rotation as euler angles");
DEFINE_bool(verbose, false, "verbose output of optimization progress");
DEFINE_int32(max_iters, 1, "max number of iterations");
DEFINE_double(min_light_change, 1E-5, "minimum change rate for light params");
DEFINE_double(min_albedo_change, 1E-5, "minimum change rate for albedo params");
DEFINE_string(out_light, "out_light.txt", "file to write light parameters");
DEFINE_string(out_image_prefix, "out_image_", "file prefix for final renders");
DEFINE_bool(use_act, false, "use activation cost regulizer");
DEFINE_double(inner_act, 1.0, "inner log scale for activation cost");
DEFINE_double(outer_act, 1.0, "outer log scale for activation cost");
DEFINE_bool(use_ref, false, "use reflectance cost regulizer");
DEFINE_double(chrom_thresh, 0.0, "chromaticiy likeness threshold for ref cost");
DEFINE_double(ref_weight, 1.0, "scaling to apply to reflectance cost");

using namespace torch;

int iteration = 0;
std::shared_ptr<Scene> scene;
std::shared_ptr<VoxelLight> light;
std::shared_ptr<MatteMaterial> material;
std::shared_ptr<Mesh> mesh;

std::vector<std::shared_ptr<Camera>> cameras;
std::vector<std::shared_ptr<Keyframe>> keyframes;

lynx::Problem* lightProblem;
VoxelCostFunction* lightCostFunction;
float* lightValues;

lynx::Problem* albedoProblem;
AlbedoCostFunction* albedoCostFunction;
float* albedoValues;

void ReadPoses(std::vector<Sophus::SE3f>& poses)
{
  poses.clear();
  std::string line;
  std::ifstream fin(FLAGS_poses);

  while(std::getline(fin, line))
  {
    Sophus::SE3f pose;

    if (FLAGS_euler)
    {
      float data[6];
      std::stringstream ss(line);
      std::string token;

      for (int j = 0; j < 6; ++j)
      {
        TORCH_ASSERT(std::getline(ss, token, ','), "invalid pose file");
        data[j] = std::stof(token);
      }

      Eigen::Quaternionf quaternion =
          Eigen::AngleAxisf(data[0], Eigen::Vector3f::UnitX()) *
          Eigen::AngleAxisf(data[1], Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(data[2], Eigen::Vector3f::UnitZ());

      pose.setQuaternion(quaternion);
      pose.translation() = Eigen::Vector3f(data[3], data[4], data[5]);
    }
    else
    {
      float data[7];
      std::stringstream ss(line);
      std::string token;

      for (int j = 0; j < 7; ++j)
      {
        TORCH_ASSERT(std::getline(ss, token, ','), "invalid pose file");
        data[j] = std::stof(token);
      }

      pose.setQuaternion(Eigen::Quaternionf(data[3], data[0], data[1], data[2]));
      pose.translation() = Eigen::Vector3f(data[4], data[5], data[6]);
    }

    poses.push_back(pose);
  }

  fin.close();
}

void WriteLightParameters()
{
  std::ofstream fout(FLAGS_out_light);
  const uint3 dims = light->GetDimensions();

  std::vector<float3> host(light->GetVoxelCount());
  optix::Buffer buffer = light->GetRadianceBuffer();
  float3* device = reinterpret_cast<float3*>(buffer->map());
  std::copy(device, device + host.size(), host.data());
  buffer->unmap();

  fout << "voxel" << std::endl;
  fout << dims.x << " " << dims.y << " " << dims.z << std::endl;

  for (const float3& radiance : host)
  {
    fout << radiance.x << " " << radiance.y << " " << radiance.z << std::endl;
  }

  fout.close();
}

void BuildScene(int argc, char** argv)
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
  light->SetRadiance(1E-8, 1E-8, 1E-8);
  scene->Add(light);

  LOG(INFO) << "Building keyframes...";

  std::vector<Sophus::SE3f> poses;
  ReadPoses(poses);
  cameras.resize(poses.size());
  keyframes.resize(cameras.size());

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    std::shared_ptr<Image> image;
    image = std::make_shared<Image>();
    image->Load(argv[i + 1]);
    image->Scale(1 / FLAGS_img_scale);

    cameras[i] = scene->CreateCamera();
    cameras[i]->SetImageSize(image->GetWidth(), image->GetHeight());
    cameras[i]->SetFocalLength(FLAGS_fx / FLAGS_img_scale, FLAGS_fy / FLAGS_img_scale);
    cameras[i]->SetCenterPoint(FLAGS_cx / FLAGS_img_scale, FLAGS_cy / FLAGS_img_scale);
    cameras[i]->SetSampleCount(FLAGS_samples);
    cameras[i]->SetTransform(poses[i]);

    // // corner04
    // cameras[i]->SetOrientation(-0.1876, 0.020051, -0.00564309, 0.982025);
    // cameras[i]->SetPosition(-0.0138542, -0.325955, 0.276384);

    // // sink01
    // cameras[i]->SetOrientation(-0.0481747, 0.00691056, -0.0414812, 0.997953);
    // cameras[i]->SetPosition(0.0237228, 0.0249558, -0.0143689);

    // // kitchen01
    // // // image_rgb_00060
    // // cameras[i]->SetOrientation(-0.0262128, 0.0620964, 0.0213342, 0.997498);
    // // cameras[i]->SetPosition(0.0173549, 0.0174267, -0.011409);
    // // image_rgb_02016
    // cameras[i]->SetOrientation(-0.0858785, -0.179851, -0.087353, 0.976037);
    // cameras[i]->SetPosition(0.607607, -0.0107669, 0.232484);

    // corner02
    // // image_rgb_01010
    // cameras[i]->SetOrientation(-0.00600349, 0.00614246, -0.0346581, 0.999362);
    // cameras[i]->SetPosition(0.0896841, 0.372875, -0.532748);
    // // image_rgb_00876
    // cameras[i]->SetOrientation(-0.00870524, -0.0240282, -0.0542144, 0.998202);
    // cameras[i]->SetPosition(0.127279, 0.321349, -0.281022);
    // image_rgb_00862
    cameras[i]->SetOrientation(-0.0256108, 0.034136, -0.0424451, 0.998187);
    cameras[i]->SetPosition(0.198602, 0.315767, -0.219588);


    Image tempImage;
    cameras[i]->CaptureNormals(tempImage);
    tempImage.Save("normal_" + std::to_string(i) + ".png");

    cameras[i]->CaptureAlbedo(tempImage);
    tempImage.Save("albedo_" + std::to_string(i) + ".png");

    keyframes[i] = std::make_shared<Keyframe>(cameras[i], image);
  }
}

void CreateLightProblem()
{
  LOG(INFO) << "Creating light problem...";

  const size_t paramCount = 3 * light->GetVoxelCount();
  optix::Buffer buffer = light->GetRadianceBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  lightValues = reinterpret_cast<float*>(pointer);

  lightProblem = new lynx::Problem();
  lightProblem->AddParameterBlock(lightValues, paramCount);
  lightProblem->SetLowerBound(lightValues, 0.0f);

  lightCostFunction = new torch::VoxelCostFunction(light);

  for (std::shared_ptr<Keyframe> keyframe : keyframes)
  {
    lightCostFunction->AddKeyframe(keyframe);
  }

  lightProblem->AddResidualBlock(lightCostFunction, nullptr, lightValues);

  if (FLAGS_use_act)
  {
    VoxelActivationCostFunction* actCostFunction;
    actCostFunction = new VoxelActivationCostFunction(light);
    actCostFunction->SetBias(1.0);
    actCostFunction->SetInnerScale(FLAGS_inner_act);
    actCostFunction->SetOuterScale(FLAGS_outer_act);
    lightProblem->AddResidualBlock(actCostFunction, nullptr, lightValues);
  }
}

void CreateAlbedoProblem()
{
  LOG(INFO) << "Creating albedo problem...";

  const size_t paramCount = 3 * material->GetAlbedoCount();
  optix::Buffer buffer = material->GetAlbedoBuffer();
  CUdeviceptr pointer = buffer->getDevicePointer(0);
  albedoValues = reinterpret_cast<float*>(pointer);

  albedoProblem = new lynx::Problem();
  albedoProblem->AddParameterBlock(albedoValues, paramCount);
  albedoProblem->SetLowerBound(albedoValues, 0.0f);
  albedoProblem->SetUpperBound(albedoValues, 1.0f);

  albedoCostFunction = new torch::AlbedoCostFunction(material, mesh);

  for (std::shared_ptr<Keyframe> keyframe : keyframes)
  {
    albedoCostFunction->AddKeyframe(keyframe);
  }

  albedoProblem->AddResidualBlock(albedoCostFunction, nullptr, albedoValues);

  if (FLAGS_use_ref)
  {
    ReflectanceCostFunction* refCostFunction;
    refCostFunction = new ReflectanceCostFunction(material, mesh);
    refCostFunction->SetChromaticityThreshold(FLAGS_chrom_thresh);
    refCostFunction->SetWeight(FLAGS_ref_weight);
    albedoProblem->AddResidualBlock(refCostFunction, nullptr, albedoValues);
  }
}

void SolveLightProblem()
{
  LOG(INFO) << "Solving light problem...";

  lynx::Solver::Options options;
  options.maxIterations = 50000;
  options.minCostChangeRate = FLAGS_min_light_change;
  options.verbose = FLAGS_verbose;

  lynx::Solver solver(lightProblem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  LOG(INFO) << std::endl << summary.BriefReport();

  const size_t count = lightProblem->GetParameterBlockSize(0);
  std::vector<Spectrum> radiance(count / 3);

  const size_t bytes = sizeof(float) * count;
  const cudaMemcpyKind type = cudaMemcpyDeviceToHost;
  LYNX_CHECK_CUDA(cudaMemcpy(radiance.data(), lightValues, bytes, type));
  light->SetRadiance(radiance);

  lightCostFunction->ClearJacobian();

  LOG(INFO) << "Rendering new light estimate...";

  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    cameras[i]->CaptureLighting(image);

    image.Save("light_estimate_" + std::to_string(i) + "_" +
        std::to_string(iteration) + ".png");
  }
}

void SolveAlbedoProblem()
{
  LOG(INFO) << "Solving albedo problem...";

  albedoCostFunction->ClearJacobian();

  lynx::Solver::Options options;
  options.maxIterations = 50000;
  options.minCostChangeRate = FLAGS_min_albedo_change;
  options.verbose = FLAGS_verbose;

  lynx::Solver solver(albedoProblem);
  solver.Configure(options);

  lynx::Solver::Summary summary;
  solver.Solve(&summary);

  LOG(INFO) << summary.BriefReport();

  const size_t count = albedoProblem->GetParameterBlockSize(0);
  std::vector<Spectrum> albedos(count / 3);

  const size_t bytes = sizeof(float) * count;
  const cudaMemcpyKind type = cudaMemcpyDeviceToHost;
  LYNX_CHECK_CUDA(cudaMemcpy(albedos.data(), albedoValues, bytes, type));
  material->SetAlbedos(albedos);

  LOG(INFO) << "Rendering new albedo estimate...";

  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    cameras[i]->CaptureAlbedo(image);

    image.Save("albedo_estimate_" + std::to_string(i) + "_" +
        std::to_string(iteration) + ".png");
  }
}

void SolveProblem(int argc, char** argv)
{
  BuildScene(argc, argv);
  // CreateAlbedoProblem();
  CreateLightProblem();

  for (iteration = 0; iteration < FLAGS_max_iters; ++iteration)
  {
    LOG(INFO) << "Starting iteration " << iteration << "...";
    // SolveAlbedoProblem();
    SolveLightProblem();
  }

  LOG(INFO) << "Saving lighting parameters...";
  WriteLightParameters();

  LOG(INFO) << "Rendering final results...";
  Image image;

  for (size_t i = 0; i < cameras.size(); ++i)
  {
    cameras[i]->Capture(image);
    image.Save(FLAGS_out_image_prefix + std::to_string(i) + ".png");
  }
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const clock_t start = clock();
  SolveProblem(argc, argv);
  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;

  LOG(INFO) << "Elapsed time: " << time;
  return 0;
}
