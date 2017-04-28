#include <torch/MeshLoader.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <torch/Exception.h>
#include <torch/Mesh.h>
#include <torch/Normal.h>
#include <torch/Point.h>

namespace torch
{

MeshLoader::MeshLoader(std::shared_ptr<Mesh> mesh) :
  m_mesh(mesh)
{
}

void MeshLoader::Load(const std::string& file)
{
  unsigned int flags = 0;
  flags |= aiProcess_Triangulate;
  flags |= aiProcess_OptimizeMeshes;
  flags |= aiProcess_JoinIdenticalVertices;
  flags |= aiProcess_RemoveRedundantMaterials;
  flags |= aiProcess_FindInstances;
  flags |= aiProcess_Debone;

  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(file.c_str(), flags);
  TORCH_ASSERT(scene, "unable to read mesh file: " + file);

  const unsigned int meshCount = scene->mNumMeshes;
  unsigned int vertexCount = 0;
  unsigned int faceCount = 0;
  bool hasNormals = true;

  for (unsigned int i = 0; i < meshCount; ++i)
  {
    const aiMesh* mesh = scene->mMeshes[i];
    vertexCount += mesh->mNumVertices;
    faceCount += mesh->mNumFaces;
    hasNormals &= mesh->HasNormals();
  }

  unsigned int normalCount = (hasNormals) ? vertexCount : 0;
  std::vector<Point> vertices(vertexCount);
  std::vector<Normal> normals(normalCount);
  std::vector<uint3> faces(faceCount);
  unsigned int vertexOffset = 0;
  unsigned int faceOffset = 0;

  for (unsigned int i = 0; i < meshCount; ++i)
  {
    const aiMesh* mesh = scene->mMeshes[i];
    const unsigned int vertexCount = mesh->mNumVertices;
    const unsigned int faceCount = mesh->mNumFaces;

    const size_t vertexBytes = sizeof(aiVector3D) * vertexCount;
    memcpy(&vertices[vertexOffset], mesh->mVertices, vertexBytes);

    if (hasNormals)
    {
      memcpy(&normals[vertexOffset], mesh->mNormals, vertexBytes);
    }

    for (unsigned int j = 0; j < faceCount; ++j)
    {
      const aiFace& face = mesh->mFaces[j];
      memcpy(&faces[faceOffset + j], face.mIndices, 3 * sizeof(unsigned int));
    }

    vertexOffset += vertexCount;
    faceOffset += faceCount;
  }

  m_mesh->SetVertices(vertices);
  m_mesh->SetNormals(normals);
  m_mesh->SetFaces(faces);
}

} // namespace torch
