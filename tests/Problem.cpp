#include <gtest/gtest.h>
#include <torch/Torch.h>
#include <Eigen/Eigen>

#include <fstream>

namespace torch
{

namespace testing
{

class Problem : public ::testing::Test
{
  public:

    Problem()
    {
      Initialize();
    }

    void CreateProblem()
    {
      m_problem = std::make_shared<torch::Problem>(m_scene,
          m_mesh, m_material, m_light, m_references);
    }

    void SetAlbedos(const std::vector<float>& albedos)
    {
      std::vector<Spectrum> spectrum(albedos.size() / 3);

      for (size_t i = 0; i < spectrum.size(); ++i)
      {
        const size_t r = 3 * i + 0;
        const size_t g = 3 * i + 1;
        const size_t b = 3 * i + 2;
        spectrum[i] = Spectrum::FromRGB(albedos[r], albedos[g], albedos[b]);
      }

      m_material->SetAlbedos(spectrum);
    }

    void SetRadiance(const std::vector<Spectrum>& radiance)
    {
      optix::Buffer buffer = m_light->GetRadianceBuffer();
      Spectrum* device = reinterpret_cast<Spectrum*>(buffer->map());
      std::copy(radiance.begin(), radiance.end(), device);
      buffer->unmap();
    }

  private:

    void Initialize()
    {
      CreateScene();
      CreateMesh();
      CreateMaterial();
      CreatePrimitive();
      CreateLight();
      CreateCameras();
      CreateImages();
      CreateReference();
    }

    void CreateScene()
    {
      m_scene = std::make_shared<Scene>();
    }

    void CreateMesh()
    {
      m_mesh = m_scene->CreateMesh();

      std::vector<Point> vertices;
      vertices.push_back(Point(-2.0, -1.5, 2));
      vertices.push_back(Point(-2.0, +1.5, 2));
      vertices.push_back(Point(+1.5, +1.25, 2));
      vertices.push_back(Point(+1.5, -1.25, 2));

      std::vector<Normal> normals;
      normals.push_back(Normal(0, 0, -1));
      normals.push_back(Normal(0, 0, -1));
      normals.push_back(Normal(0, 0, -1));
      normals.push_back(Normal(0, 0, -1));

      std::vector<uint3> faces;
      faces.push_back(make_uint3(0, 1, 2));
      faces.push_back(make_uint3(0, 3, 2));

      m_mesh->SetVertices(vertices);
      m_mesh->SetNormals(normals);
      m_mesh->SetFaces(faces);
    }

    void CreateMaterial()
    {
      m_material = m_scene->CreateMatteMaterial();

      std::vector<Spectrum> albedos;
      albedos.push_back(Spectrum::FromRGB(0.5, 0.1, 0.2));
      albedos.push_back(Spectrum::FromRGB(0.4, 0.8, 0.3));
      albedos.push_back(Spectrum::FromRGB(0.8, 0.2, 0.7));
      albedos.push_back(Spectrum::FromRGB(0.2, 0.8, 0.1));

      m_material->SetAlbedos(albedos);
    }

    void CreatePrimitive()
    {
      m_primitive = m_scene->CreatePrimitive();
      m_primitive->SetGeometry(m_mesh);
      m_primitive->SetMaterial(m_material);
      m_scene->Add(m_primitive);
    }

    void CreateLight()
    {
      m_light = m_scene->CreateEnvironmentLight();
      m_light->SetRowCount(4);
      m_light->SetRadiance(0.01, 0.01, 0.01);
      m_scene->Add(m_light);
    }

    void CreateCameras()
    {
      std::shared_ptr<Camera> camera;
      camera = m_scene->CreateCamera();
      camera->SetOrientation(0, 0, 0);
      camera->SetPosition(0, 0, 0);
      camera->SetImageSize(160, 120);
      camera->SetFocalLength(80, 80);
      camera->SetCenterPoint(80, 60);
      camera->SetSampleCount(6);
      m_cameras.push_back(camera);
    }

    void CreateImages()
    {
      std::shared_ptr<Image> image;
      image = std::make_shared<Image>();
      // TODO: resize image
      // TODO: populate image
      m_images.push_back(image);
    }

    void CreateReference()
    {
      size_t size = std::min(m_cameras.size(), m_images.size());
      m_references.resize(size);

      for (size_t i = 0; i < size; ++i)
      {
        std::shared_ptr<Camera> cam = m_cameras[i];
        std::shared_ptr<Image> img = m_images[i];
        std::shared_ptr<ReferenceImage> reference;
        reference = std::make_shared<ReferenceImage>(cam, img);
        m_references[i] = reference;
      }
    }

  protected:

    std::shared_ptr<Scene> m_scene;

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    std::shared_ptr<Primitive> m_primitive;

    std::shared_ptr<EnvironmentLight> m_light;

    std::vector<std::shared_ptr<Camera>> m_cameras;

    std::vector<std::shared_ptr<Image>> m_images;

    std::vector<std::shared_ptr<ReferenceImage>> m_references;

    std::shared_ptr<torch::Problem> m_problem;
};

float3 GetSparseValue(unsigned int row, unsigned int col,
  const std::vector<float3>& values,
  const std::vector<unsigned int>& rowOffsets,
  const std::vector<unsigned int>& colIndices)
{
  unsigned int begin = rowOffsets[row];
  unsigned int end = rowOffsets[row + 1];

  if (begin < end)
  {
    unsigned int index = begin;

    while (begin < end)
    {
      index = (begin + end) / 2;

      if (col == colIndices[index])
      {
        return values[index];
      }

      (col < colIndices[index]) ? end = index : begin = index + 1;
    }
  }

  return make_float3(0, 0, 0);
}

TEST_F(Problem, AlbedoDerivatives)
{
  CreateProblem();

  std::vector<uint2> validPixels;
  m_references[0]->GetValidPixels(validPixels);

  std::vector<float> channels =
  {
    0.5, 0.1, 0.2,
    0.4, 0.8, 0.3,
    0.8, 0.2, 0.7,
    0.2, 0.8, 0.1
  };

  SetAlbedos(channels);
  m_problem->ComputeAlbedoDerivatives();

  std::vector<float3> r0;
  m_problem->GetRenderValues(r0);

  std::shared_ptr<SparseMatrix> jacobian;
  jacobian = m_problem->GetAlbedoJacobian(0);

  std::vector<float3> values;
  std::vector<unsigned int> rowOffsets;
  std::vector<unsigned int> colIndices;
  jacobian->GetValues(values);
  jacobian->GetRowOffsets(rowOffsets);
  jacobian->GetColumnIndices(colIndices);

  const uint rowCount = channels.size();
  const uint colCount = 3 * m_references[0]->GetValidPixelCount();
  Eigen::MatrixXf analJacobian(rowCount, colCount);
  analJacobian.setZero();

  for (uint col = 0; col < colCount / 3; ++col)
  {
    for (uint row = 0; row < rowCount / 3; ++row)
    {
      const float3 value = GetSparseValue(row, col, values,
          rowOffsets, colIndices);

      analJacobian(3 * row + 0, 3 * col + 0) = value.x;
      analJacobian(3 * row + 1, 3 * col + 1) = value.y;
      analJacobian(3 * row + 2, 3 * col + 2) = value.z;
    }
  }

  const float delta = 0.05;
  Eigen::MatrixXf finiteJacobian(rowCount, colCount);
  finiteJacobian.setZero();

  for (size_t i = 0; i < channels.size(); ++i)
  {
    std::cout << (i + 1) << " / " << channels.size() << std::endl;

    const float orig = channels[i];
    channels[i] += delta;
    SetAlbedos(channels);
    m_problem->ComputeAlbedoDerivatives();

    std::vector<float3> r1;
    m_problem->GetRenderValues(r1);

    for (size_t col = 0; col < r1.size(); ++col)
    {
      r1[col] = (r1[col] - r0[col]) / delta;
      finiteJacobian(i, 3 * col + 0) = r1[col].x;
      finiteJacobian(i, 3 * col + 1) = r1[col].y;
      finiteJacobian(i, 3 * col + 2) = r1[col].z;
    }

    channels[i] = orig;
  }

  for (uint col = 0; col < colCount; ++col)
  {
    for (uint row = 0; row < rowCount; ++row)
    {
      const float expected = finiteJacobian(row, col);
      const float found = analJacobian(row, col);
      ASSERT_NEAR(expected, found, 1E-4);
    }
  }
}

TEST_F(Problem, LightDerivatives)
{
  CreateProblem();

  const unsigned int dirCount = m_light->GetDirectionCount();
  std::vector<Spectrum> radiance(dirCount);
  const Spectrum defaultRadiance = Spectrum::FromRGB(0.1, 0.1, 0.1);
  std::fill(radiance.begin(), radiance.end(), defaultRadiance);

  const unsigned int paramCount = 3 * dirCount;
  float* parameters = reinterpret_cast<float*>(radiance.data());

  m_problem->ComputeLightDerivatives();
  SetRadiance(radiance);
  m_problem->ComputeLightDerivatives();

  std::vector<float3> r0;
  m_problem->GetRenderValues(r0);

  optix::Buffer lightDerivs;
  lightDerivs = m_problem->GetLightDerivativeBuffer();

  RTsize colCount, rowCount;
  lightDerivs->getSize(colCount, rowCount);

  Eigen::MatrixXf analJacobian(paramCount, 3 * r0.size());
  analJacobian.setZero();

  float3* device = reinterpret_cast<float3*>(lightDerivs->map());
  size_t devIndex = 0;

  std::cout << "Eigen Rows: " << analJacobian.rows() << ", Cols: " <<  analJacobian.cols() << std::endl;
  std::cout << "Optix Rows: " << rowCount << ", Cols: " <<  colCount << std::endl;

  for (uint row = 0; row < rowCount; ++row)
  {
    for (uint col = 0; col < colCount; ++col)
    {
      analJacobian(3 * col + 0, 3 * row + 0) = device[devIndex].x;
      analJacobian(3 * col + 1, 3 * row + 1) = device[devIndex].y;
      analJacobian(3 * col + 2, 3 * row + 2) = device[devIndex].z;
      ++devIndex;
    }
  }

  lightDerivs->unmap();

  Eigen::MatrixXf finiteJacobian(paramCount, 3 * r0.size());
  finiteJacobian.setZero();

  const float delta = 0.05;

  for (unsigned int i = 0; i < paramCount; ++i)
  {
    std::cout << (i + 1) << " / " << paramCount << std::endl;

    const float orig = parameters[i];
    parameters[i] += delta;

    SetRadiance(radiance);
    m_problem->ComputeLightDerivatives();

    std::vector<float3> r1;
    m_problem->GetRenderValues(r1);

    for (size_t col = 0; col < r1.size(); ++col)
    {
      r1[col] = (r1[col] - r0[col]) / delta;
      finiteJacobian(i, 3 * col + 0) = r1[col].x;
      finiteJacobian(i, 3 * col + 1) = r1[col].y;
      finiteJacobian(i, 3 * col + 2) = r1[col].z;
    }

    parameters[i] = orig;
  }

  for (uint col = 0; col < finiteJacobian.cols(); ++col)
  {
    for (uint row = 0; row < finiteJacobian.rows(); ++row)
    {
      const float expected = finiteJacobian(row, col);
      const float found = analJacobian(row, col);
      ASSERT_NEAR(expected, found, 1E-4);
    }
  }
}

} // namespace testing

} // namespace torch