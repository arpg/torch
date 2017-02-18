#include <torch/MeshWriter.h>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <torch/Exception.h>
#include <torch/MatteMaterial.h>
#include <torch/Mesh.h>
#include <torch/Normal.h>
#include <torch/Point.h>
#include <torch/Spectrum.h>

#include <iostream>

namespace torch
{

MeshWriter::MeshWriter(std::shared_ptr<Mesh> mesh,
    std::shared_ptr<MatteMaterial> material) :
  m_mesh(mesh),
  m_material(material)
{
  Initialize();
}

void MeshWriter::Write(const std::string& file)
{
  Assimp::Exporter exporter;
  aiReturn result = exporter.Export(m_aiScene, "ply", file);
  if (result != aiReturn_SUCCESS) throw Exception(exporter.GetErrorString());
}

void MeshWriter::Initialize()
{
  CreateMesh();
  AddVertices();
  AddNormals();
  AddFaces();
  AddColors();
}

void MeshWriter::CreateMesh()
{
  m_aiMesh = new aiMesh();
  m_aiMesh->mMaterialIndex = 0;

  m_aiScene = new aiScene();
  m_aiScene->mMeshes = new aiMesh*[1];
  m_aiScene->mMeshes[0] = m_aiMesh;
  m_aiScene->mNumMeshes = 1;

  m_aiScene->mMaterials = new aiMaterial*[1];
  m_aiScene->mMaterials[0] = new aiMaterial();
  m_aiScene->mNumMaterials = 1;

  m_aiScene->mRootNode = new aiNode();
  m_aiScene->mRootNode->mMeshes = new unsigned int [1];
  m_aiScene->mRootNode->mMeshes[0] = 0;
  m_aiScene->mRootNode->mNumMeshes = 1;
}

void MeshWriter::AddVertices()
{
  std::vector<Point> vertices;
  m_mesh->GetVertices(vertices);
  m_aiMesh->mNumVertices = vertices.size();
  m_aiMesh->mVertices = new aiVector3D[vertices.size()];

  for (size_t i = 0; i < vertices.size(); ++i)
  {
    const Point& v = vertices[i];
    m_aiMesh->mVertices[i] = aiVector3D(v.x, v.y, v.z);
  }
}

void MeshWriter::AddNormals()
{
  std::vector<Normal> normals;
  m_mesh->GetNormals(normals);
  m_aiMesh->mNormals = new aiVector3D[normals.size()];

  for (size_t i = 0; i < normals.size(); ++i)
  {
    const Normal& n = normals[i];
    m_aiMesh->mNormals[i] = aiVector3D(n.x, n.y, n.z);
  }
}

void MeshWriter::AddFaces()
{
  std::vector<uint3> faces;
  m_mesh->GetFaces(faces);
  m_aiMesh->mNumFaces = faces.size();
  m_aiMesh->mFaces = new aiFace[faces.size()];

  for (size_t i = 0; i < faces.size(); ++i)
  {
    m_aiMesh->mFaces[i].mIndices = new unsigned int[3];
    m_aiMesh->mFaces[i].mIndices[0] = faces[i].x;
    m_aiMesh->mFaces[i].mIndices[1] = faces[i].y;
    m_aiMesh->mFaces[i].mIndices[2] = faces[i].z;
    m_aiMesh->mFaces[i].mNumIndices = 3;
  }
}

void MeshWriter::AddColors()
{
  std::vector<Spectrum> albedos;
  m_material->GetAlbedos(albedos);
  m_aiMesh->mColors[0] = new aiColor4D[albedos.size()];

  for (size_t i = 0; i < albedos.size(); ++i)
  {
    const Vector& a = albedos[i].GetRGB();
    m_aiMesh->mColors[0][i] = aiColor4D(a.x, a.y, a.z, 1);
  }
}

} // namespace torch