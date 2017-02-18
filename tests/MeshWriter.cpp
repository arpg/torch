#include <gtest/gtest.h>
#include <torch/Torch.h>

namespace torch
{
namespace testing
{

TEST(MeshWriter, General)
{
  const std::string file("mesh.ply");

  std::vector<Point> outVertices;
  outVertices.push_back(Point( 0.0,  0.0,  0.0));
  outVertices.push_back(Point( 1.0, -1.0,  0.5));
  outVertices.push_back(Point( 1.0,  1.0,  0.5));
  outVertices.push_back(Point(-1.0,  1.0,  0.5));
  outVertices.push_back(Point(-1.0, -1.0,  0.5));

  std::vector<Normal> outNormals;
  outNormals.push_back(Normal( 0,  0, -1));
  outNormals.push_back(Normal( 0,  0, -1));
  outNormals.push_back(Normal( 0,  0, -1));
  outNormals.push_back(Normal( 0,  0, -1));
  outNormals.push_back(Normal( 0,  0, -1));

  std::vector<uint3> outFaces;
  outFaces.push_back(make_uint3(0, 1, 2));
  outFaces.push_back(make_uint3(0, 2, 3));
  outFaces.push_back(make_uint3(0, 3, 4));
  outFaces.push_back(make_uint3(0, 4, 1));

  std::vector<Spectrum> outAlbedos;
  outAlbedos.push_back(Spectrum::FromRGB(0.65, 0.15, 0.15));
  outAlbedos.push_back(Spectrum::FromRGB(0.14, 0.58, 0.14));
  outAlbedos.push_back(Spectrum::FromRGB(0.23, 0.83, 0.52));
  outAlbedos.push_back(Spectrum::FromRGB(0.52, 0.52, 0.33));
  outAlbedos.push_back(Spectrum::FromRGB(0.81, 0.11, 0.91));

  Scene scene;

  std::shared_ptr<Mesh> outMesh;
  outMesh = scene.CreateMesh();
  outMesh->SetVertices(outVertices);
  outMesh->SetNormals(outNormals);
  outMesh->SetFaces(outFaces);

  std::shared_ptr<MatteMaterial> outMaterial;
  outMaterial = scene.CreateMatteMaterial();
  outMaterial->SetAlbedos(outAlbedos);

  MeshWriter writer(outMesh, outMaterial);
  writer.Write(file);

  std::shared_ptr<Mesh> inMesh;
  inMesh = scene.CreateMesh();

  MeshLoader loader(inMesh);
  loader.Load(file);

  MaterialLoader materialLoader(inMesh->GetContext());

  std::shared_ptr<MatteMaterial> inMaterial =
      std::static_pointer_cast<MatteMaterial>(materialLoader.Load(file));

  std::vector<Point> inVertices;
  inMesh->GetVertices(inVertices);

  ASSERT_EQ(outVertices.size(), inVertices.size());

  for (size_t i = 0; i < outVertices.size(); ++i)
  {
    ASSERT_NEAR(outVertices[i].x, inVertices[i].x, 1E-6);
    ASSERT_NEAR(outVertices[i].y, inVertices[i].y, 1E-6);
    ASSERT_NEAR(outVertices[i].z, inVertices[i].z, 1E-6);
  }

  std::vector<Normal> inNormals;
  inMesh->GetNormals(inNormals);

  ASSERT_EQ(outNormals.size(), inNormals.size());

  for (size_t i = 0; i < outNormals.size(); ++i)
  {
    ASSERT_NEAR(outNormals[i].x, inNormals[i].x, 1E-6);
    ASSERT_NEAR(outNormals[i].y, inNormals[i].y, 1E-6);
    ASSERT_NEAR(outNormals[i].z, inNormals[i].z, 1E-6);
  }

  std::vector<uint3> inFaces;
  inMesh->GetFaces(inFaces);

  ASSERT_EQ(outFaces.size(), inFaces.size());

  for (size_t i = 0; i < outFaces.size(); ++i)
  {
    ASSERT_EQ(outFaces[i].x, inFaces[i].x);
    ASSERT_EQ(outFaces[i].y, inFaces[i].y);
    ASSERT_EQ(outFaces[i].z, inFaces[i].z);
  }

  std::vector<Spectrum> inAlbedos;
  inMaterial->GetAlbedos(inAlbedos);

  ASSERT_EQ(outAlbedos.size(), inAlbedos.size());

  for (size_t i = 0; i < outAlbedos.size(); ++i)
  {
    const Vector outAlbedo = outAlbedos[i].GetRGB();
    const Vector inAlbedo = outAlbedos[i].GetRGB();
    ASSERT_NEAR(outAlbedo.x, inAlbedo.x, 1E-6);
    ASSERT_NEAR(outAlbedo.y, inAlbedo.y, 1E-6);
    ASSERT_NEAR(outAlbedo.z, inAlbedo.z, 1E-6);
  }
}

} // namespace testing

} // namespace torch