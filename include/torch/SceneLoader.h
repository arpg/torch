#pragma once

#include <unordered_map>
#include <tinyxml2.h>
#include <torch/Core.h>

namespace torch
{

class SceneLoader
{
  protected:

    typedef tinyxml2::XMLElement Element;

    typedef std::shared_ptr<Node>(SceneLoader::*NodeParser)(Element*);

    typedef std::shared_ptr<Geometry>(SceneLoader::*GeometryParser)(Element*);

    typedef std::shared_ptr<Material>(SceneLoader::*MaterialParser)(Element*);

  public:

    SceneLoader(const std::string& file);

    void Load(Scene& scene);

  protected:

    void ParseScene(Element* elem);

    std::shared_ptr<Node> ParseNode(Element* elem);

    std::shared_ptr<Node> ParseCamera(Element* elem);

    std::shared_ptr<Node> ParseGroup(Element* elem);

    std::shared_ptr<Node> ParseAreaLight(Element* elem);

    std::shared_ptr<Node> ParseDirectionalLight(Element* elem);

    std::shared_ptr<Node> ParseEnvironmentLight(Element* elem);

    std::shared_ptr<Node> ParsePointLight(Element* elem);

    std::shared_ptr<Node> ParsePrimitive(Element* elem);

    std::shared_ptr<Geometry> ParseGeometry(Element* elem);

    std::shared_ptr<Geometry> ParseGeometryGroup(Element* elem);

    std::shared_ptr<Geometry> ParseMesh(Element* elem);

    std::shared_ptr<Geometry> ParseSphere(Element* elem);

    std::shared_ptr<Material> ParseMaterial(Element* elem);

    std::shared_ptr<Material> ParseMatteMaterial(Element* elem);

    Spectrum ParseSpectrum(Element* elem);

    Transform ParseTransform(Element* elem);

    Vector ParseOrientation(Element* elem);

    Vector ParsePosition(Element* elem);

    Vector ParseScale(Element* elem);

    Vector ParseVector(Element* elem);

    unsigned int ParseUnsignedInt(Element* elem);

    void ParseUint2(Element* elem, unsigned int& x, unsigned int& y);

    void ParseFloat2(Element* elem, float& x, float& y);

    bool IsNode(Element* elem);

    bool IsGeometry(Element* elem);

    bool IsMaterial(Element* elem);

  private:

    void Initialize();

    void LoadFile();

    void CreateNodeParsers();

    void CreateGeometryParsers();

    void CreateMaterialParsers();

  protected:

    Scene* m_scene;

    const std::string m_file;

    tinyxml2::XMLDocument m_doc;

    std::unordered_map<std::string, NodeParser> m_nodeParsers;

    std::unordered_map<std::string, GeometryParser> m_geometryParsers;

    std::unordered_map<std::string, MaterialParser> m_materialParsers;
};

} // namespace torch