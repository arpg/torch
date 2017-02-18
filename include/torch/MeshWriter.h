#pragma once

#include <torch/Core.h>

class aiScene;
class aiMesh;

namespace torch
{

class MeshWriter
{
  public:

    MeshWriter(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<MatteMaterial> material);

    void Write(const std::string& file);

  private:

    void Initialize();

    void CreateMesh();

    void AddVertices();

    void AddNormals();

    void AddFaces();

    void AddColors();

  protected:

    std::shared_ptr<Mesh> m_mesh;

    std::shared_ptr<MatteMaterial> m_material;

    aiScene* m_aiScene;

    aiMesh* m_aiMesh;
};

} // namespace torch