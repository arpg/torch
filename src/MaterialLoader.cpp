#include <torch/MaterialLoader.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <torch/Context.h>
#include <torch/Exception.h>
#include <torch/MatteMaterial.h>
#include <torch/Spectrum.h>

namespace torch
{

MaterialLoader::MaterialLoader(std::shared_ptr<Context> context) :
  m_context(context)
{
}

std::shared_ptr<Material> MaterialLoader::Load(const std::string& file)
{
  std::shared_ptr<MatteMaterial> material;
  material = std::make_shared<MatteMaterial>(m_context);
  m_context->RegisterObject(material);

  unsigned int flags = 0;
  flags |= aiProcess_Triangulate;
  flags |= aiProcess_OptimizeMeshes;
  flags |= aiProcess_JoinIdenticalVertices;
  flags |= aiProcess_RemoveRedundantMaterials;
  flags |= aiProcess_FindInstances;
  flags |= aiProcess_Debone;

  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(file.c_str(), flags);
  TORCH_ASSERT(scene, "unable to read material file: " + file);

  const unsigned int meshCount = scene->mNumMeshes;
  unsigned int colorCount = 0;
  bool hasColors = true;

  for (unsigned int i = 0; i < meshCount; ++i)
  {
    const aiMesh* mesh = scene->mMeshes[i];
    colorCount += mesh->mNumVertices;
    hasColors &= mesh->HasVertexColors(0);
    if (!hasColors) return material;
  }

  std::vector<Spectrum> colors(colorCount);
  unsigned int colorIndex = 0;

  for (unsigned int i = 0; i < meshCount; ++i)
  {
    const aiMesh* mesh = scene->mMeshes[i];
    const unsigned int meshColorCount = mesh->mNumVertices;
    aiColor4D* colorSet = mesh->mColors[0];

    for (unsigned int j = 0; j < meshColorCount; ++j)
    {
      const aiColor4D& color = colorSet[j];
      colors[colorIndex++] = Spectrum::FromRGB(color.r, color.g, color.b);
    }
  }

  material->SetAlbedos(colors);
  return material;
}

} // namespace torch