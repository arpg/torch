#include <gtest/gtest.h>
#include <torch/Torch.h>

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
      vertices.push_back(Point(+2.0, +1.5, 2));
      vertices.push_back(Point(+2.0, -1.5, 2));

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
      m_light->SetRowCount(21);
      m_light->SetRadiance(1E-4, 1E-4, 1E-4);
    }

    void CreateCameras()
    {
      std::shared_ptr<Camera> camera;
      camera = m_scene->CreateCamera();
      camera->SetOrientation(0, 0, 0);
      camera->SetPosition(0, 0, 0);
      camera->SetImageSize(320, 240);
      camera->SetFocalLength(160, 160);
      camera->SetCenterPoint(160, 120);
      camera->SetSampleCount(4);
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

TEST_F(Problem, AlbedoDerivatives)
{
  CreateProblem();
  // m_problem->ComputeAlbedoDerivatives();

  // Problem problem;
  // problem.ComputeAlbedoDerivatives();
  // std::shared_ptr<SparseMatrix> jacobian;
  // jacobian = problem.GetAlbedoJacobian(0);
}

} // namespace testing

} // namespace torch