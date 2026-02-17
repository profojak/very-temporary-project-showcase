#pragma once

#include "Primitives.h"
#include "../Utilities/FastNoiseLite.h"

#include <vector>

/*
 * Y = up
 * X = right
 * Z = forward
 */

#define CHUNK_SIZE_X 16
#define CHUNK_SIZE_Z 16
#define CHUNK_SIZE_Y 128

class VertexBuffer {
public:
    VertexBuffer() = delete;
    VertexBuffer(const Device& m_device, const void* m_vertexData, UINT sizeofVertices);
    ~VertexBuffer();

    ID3D11Buffer* GetBuffer() { return m_vertexBuffer; }
    ID3D11Buffer** GetBufferPtr() { return &m_vertexBuffer; }

private:
    ID3D11Buffer* m_vertexBuffer;
};

struct VoxelInstance {
    FLOAT x, y, z;
    UINT dir;
    UINT scale_x, scale_y;
};

struct VoxelVertex {
    UINT id;
    FLOAT color[4];
};

class VoxelChunk : public Mesh {
    typedef Mesh Super;

    struct Geometry {
        std::vector<VoxelInstance> m_instances;

        ID3D11Buffer* m_vertexBuffer = NULL;
        ID3D11Buffer* m_instanceBuffer = NULL;

        bool IsEmpty() const { return m_instances.empty(); }
    };

    enum EFaceDirection : uint8_t {
        Left  = 0,
        Right = 1,
        Down  = 2,
        Up    = 3,
        Front = 4,
        Back  = 5,
    };

public:
    VoxelChunk();
    ~VoxelChunk();

    const Vertex* GetVertices() const override;
    UINT GetVerticesSize() const override;
    const WORD* GetIndices() const override;
    UINT GetIndicesSize() const override;

    int ToLinearCoords(int x, int y, int z) const;
    Vector3<int> FromLinearCoords(int id) const;
    short GetVoxelValue(int x, int y, int z) const;

    void Initialize(Device& device) override;
    void Draw(Device& device, Camera& camera, float lod_dist) override;
    void DrawGeometry(const Geometry& geometry, Device& device) const;

    void AddVoxelGeometry(Geometry& geometry, Vector3<float> relativePos,
        UINT scale_x, UINT scale_y, EFaceDirection faceDir);
    void InitializeGeometry();
    VertexBuffer* GetVertexBuffer(const Device& device);

private:
    FastNoiseLite noiseGenerator;

    static const VoxelVertex m_faceVerts[];

    struct LOD_Geometry {
        Geometry geometries[6];
    };

    std::vector<LOD_Geometry> m_lod_geometries{ 4 };
};
 
inline const VoxelVertex VoxelChunk::m_faceVerts[] = {
    { 0, { 1.0f, 0.0f, 0.0f, 1.0f }}, // 0
    { 1, { 0.0f, 1.0f, 0.0f, 1.0f }}, // 1
    { 2, { 1.0f, 1.0f, 0.0f, 1.0f }}, // 3
    { 3, { 0.0f, 0.0f, 1.0f, 1.0f }}, // 2
};
