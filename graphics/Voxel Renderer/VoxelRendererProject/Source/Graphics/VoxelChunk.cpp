#include "VoxelChunk.h"

#include <ios>
#include <functional>

#include "Renderer.h"
#include "../Utilities/Logger.h"
#include "../Utilities/ErrorHandler.h"

VertexBuffer::VertexBuffer(const Device& device, const void* m_vertexData, UINT sizeofVertices) {
    // Create vertex buffer
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-createbuffer
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeofVertices;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    bd.StructureByteStride = 0;

    D3D11_SUBRESOURCE_DATA vertexData;
    ZeroMemory(&vertexData, sizeof(vertexData));
    vertexData.pSysMem = m_vertexData;
    vertexData.SysMemPitch = 0;
    vertexData.SysMemSlicePitch = 0;

    HRESULT hr = device.GetDevice()->CreateBuffer(&bd, &vertexData, &m_vertexBuffer);
    std::stringstream ss;
    ss << std::hex << static_cast<unsigned long>(hr);
    ThrowIfFailed(hr, "Failed to create mesh vertex buffer! Error code: ", ss.str());
}

VertexBuffer::~VertexBuffer() {
    if (m_vertexBuffer)
        m_vertexBuffer->Release();
}

VoxelChunk::VoxelChunk() : Mesh() {
    noiseGenerator.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noiseGenerator.SetFrequency(0.01f);
    noiseGenerator.SetFractalType(FastNoiseLite::FractalType_FBm);
    noiseGenerator.SetFractalOctaves(6);
    noiseGenerator.SetFractalLacunarity(1.94f);
    noiseGenerator.SetFractalGain(0.46f);
    noiseGenerator.SetFractalWeightedStrength(0.38f)
;}

VoxelChunk::~VoxelChunk() {
    int numGeometries = sizeof(m_lod_geometries[0].geometries) / sizeof(Geometry*);
    for (size_t j = 0; j < m_lod_geometries.size(); j++) {
        for (int i = 0; i < numGeometries; i++) {
            m_lod_geometries[j].geometries[i].m_vertexBuffer->Release();
            m_lod_geometries[j].geometries[i].m_instanceBuffer->Release();
        }
    }
}

const Vertex* VoxelChunk::GetVertices() const {
    return nullptr;
}

UINT VoxelChunk::GetVerticesSize() const {
    return 0;
}

const WORD* VoxelChunk::GetIndices() const {
    return nullptr;
}

UINT VoxelChunk::GetIndicesSize() const {
    return 0;
}

int VoxelChunk::ToLinearCoords(int x, int y, int z) const {
    return y + x * CHUNK_SIZE_Y + z * CHUNK_SIZE_Y * CHUNK_SIZE_X;
}

Vector3<int> VoxelChunk::FromLinearCoords(int id) const {
    int z = id / (CHUNK_SIZE_Y * CHUNK_SIZE_X);
    id %= (CHUNK_SIZE_Y * CHUNK_SIZE_X);
    int x = id / CHUNK_SIZE_Y;
    int y = id % CHUNK_SIZE_Y;
    return { x, y, z };
}

short VoxelChunk::GetVoxelValue(int x, int y, int z) const {
    float blockPosX = (float)x + transform.Position.x;
    float blockPosZ = (float)z + transform.Position.z;
    float noise = noiseGenerator.GetNoise(blockPosX, blockPosZ);
    noise = (noise + 1.0f) / 2.0f;
    noise *= CHUNK_SIZE_Y;
    return (float)y > noise ? 0 : 1;
}

void VoxelChunk::AddVoxelGeometry(Geometry& geometry, Vector3<float> relativePos,
    UINT scale_x, UINT scale_y, EFaceDirection faceDir) {
    VoxelInstance instance{};
    instance.x = relativePos.x;
    instance.y = relativePos.y;
    instance.z = relativePos.z;
    instance.dir = faceDir;
    instance.scale_x = scale_x;
    instance.scale_y = scale_y;
    geometry.m_instances.push_back(instance);
}

void VoxelChunk::InitializeGeometry() {
    for (size_t lod_id = 0; lod_id < m_lod_geometries.size(); lod_id++) {
        auto& lod_geometry = m_lod_geometries[lod_id];

        int step = int(pow(2, lod_id));
        int size_x = CHUNK_SIZE_X / step;
        int size_y = CHUNK_SIZE_Y / step;
        int seize_z = CHUNK_SIZE_Z / step;

        for (auto& g : lod_geometry.geometries)
            g.m_instances.clear();

        for (int bx = 0; bx < size_x; ++bx) {
            for (int bz = 0; bz < seize_z; ++bz) {
                for (int by = 0; by < size_y; ++by) {
                    float world_x = transform.Position.x + bx * step;
                    float world_z = transform.Position.z + bz * step;

                    // Check if current NxNxN voxel is solid (it has at least one 1x1x1 voxel inside)
                    bool solid = false;
                    for (int ox = 0; ox < step && !solid; ++ox)
                        for (int oy = 0; oy < step && !solid; ++oy)
                            for (int oz = 0; oz < step && !solid; ++oz)
                                if (GetVoxelValue(bx * step + ox, by * step + oy, bz * step + oz) != 0)
                                    solid = true;
                    if (!solid)
                        continue;

                    // Reject current NxNxN voxel if under ground and not visible
                    float terrain_height = (noiseGenerator.GetNoise(world_x, world_z) + 1.0f)
                        * 0.5f * CHUNK_SIZE_Y + transform.Position.y;
                    float voxel_top_y = transform.Position.y + (by + 1) * step;
                    if (voxel_top_y <= terrain_height) {
                        static const Vector2<int> offsets[4] = { {1, 0}, {0, 1}, {-1, 0}, {0, -1} };
                        bool visible = false;
                        for (int f = 0; f < 4; ++f) {
                            terrain_height = (noiseGenerator.GetNoise(world_x + offsets[f].x * step,
                                world_z + offsets[f].y * step) + 1.0f) * 0.5f * CHUNK_SIZE_Y + transform.Position.y;
                            if (terrain_height <= voxel_top_y) {
                                visible = true;
                                break;
                            }
                        }
                        if (!visible)
                            continue;
                    }

                    // Add only those faces that are not obscured by neighbor voxels
                    static const Vector3<int> offsets[6] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
                    for (int f = 0; f < 6; ++f) {
                        int nbx = bx + offsets[f].x;
                        int nby = by + offsets[f].y;
                        int nbz = bz + offsets[f].z;
                        solid = false;
                        if (nbx >= 0 && nbx < size_x && nby >= 0 && nby < size_y && nbz >= 0 && nbz < seize_z)
                            for (int ox = 0; ox < step && !solid; ++ox)
                                for (int oy = 0; oy < step && !solid; ++oy)
                                    for (int oz = 0; oz < step && !solid; ++oz)
                                        if (GetVoxelValue(nbx * step + ox, nby * step + oy, nbz * step + oz) != 0)
                                            solid = true;

                        if (!solid) {
                            Vector3<float> pos{
                                bx * step + 0.5f * step,
                                by * step + 0.5f * step,
                                bz * step + 0.5f * step
                            };
                            AddVoxelGeometry(lod_geometry.geometries[f], pos, step, step, EFaceDirection(f));
                        }
                    }
                }
            }
        }
    }
}

VertexBuffer* VoxelChunk::GetVertexBuffer(const Device& device) {
    static VertexBuffer vertexBuffer(device, m_faceVerts, sizeof(m_faceVerts));
    return &vertexBuffer;
}

void VoxelChunk::Initialize(Device& device) {
    InitializeGeometry();

    int numGeometries = sizeof(m_lod_geometries[0].geometries[0]) / sizeof(Geometry*);
    for (size_t lod_id = 0; lod_id < m_lod_geometries.size(); lod_id++) {
        for (int i = 0; i < numGeometries; i++) {
            Geometry& geometry = m_lod_geometries[lod_id].geometries[i];
            if (geometry.IsEmpty())
                continue;

            const VoxelInstance* m_instances = geometry.m_instances.data();
            const UINT sizeofInstances = (UINT)geometry.m_instances.size() * sizeof(VoxelInstance);

            ThrowIfFailed(sizeofInstances > 0, "Refused to create zero-byte buffer!");
            ThrowIfFailed(m_instances != nullptr, "VoxelInstance is null!");

            // Create vertex buffer
            geometry.m_vertexBuffer = GetVertexBuffer(device)->GetBuffer();

            // Create instance buffer
            D3D11_BUFFER_DESC instBufDesc;
            ZeroMemory(&instBufDesc, sizeof(instBufDesc));
            instBufDesc.Usage = D3D11_USAGE_DEFAULT;
            instBufDesc.ByteWidth = sizeofInstances;
            instBufDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            instBufDesc.CPUAccessFlags = 0;
            instBufDesc.MiscFlags = 0;
            instBufDesc.StructureByteStride = 0;

            D3D11_SUBRESOURCE_DATA instanceData;
            ZeroMemory(&instanceData, sizeof(instanceData));
            instanceData.pSysMem = m_instances;
            instanceData.SysMemPitch = 0;
            instanceData.SysMemSlicePitch = 0;

            HRESULT hr = device.GetDevice()->CreateBuffer(&instBufDesc, &instanceData, &geometry.m_instanceBuffer);
            std::stringstream ss;
            ss << std::hex << static_cast<unsigned long>(hr);
            ThrowIfFailed(hr, "Failed to create mesh instance buffer! Error code: ", ss.str());
        }
    }
}

void VoxelChunk::DrawGeometry(const Geometry& geometry, Device& device) const {
    if (geometry.IsEmpty())
        return;

    UINT strides[2] = { sizeof(VoxelVertex), sizeof(VoxelInstance) };
    UINT offsets[2] = { 0, 0 };
    ID3D11Buffer* buffers[2] = { geometry.m_vertexBuffer, geometry.m_instanceBuffer };

    UINT numInstances = (UINT)geometry.m_instances.size();
    device.GetContext()->IASetVertexBuffers(0, 2, buffers, strides, offsets);
    device.GetContext()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    device.GetContext()->DrawInstanced(4, numInstances, 0, 0);
}

void VoxelChunk::Draw(Device& device, Camera& camera, float lod_dist) {
    DirectX::XMVECTOR& camPos = camera.m_position;

    int idCamX = (int)floor(DirectX::XMVectorGetX(camPos) / CHUNK_SIZE_X);
    int idCamZ = (int)floor(DirectX::XMVectorGetZ(camPos) / CHUNK_SIZE_Z);

    int idChunkX = (int)floor(transform.Position.x / CHUNK_SIZE_X);
    int idChunkZ = (int)floor(transform.Position.z / CHUNK_SIZE_Z);

    float dist = DirectX::XMVectorGetX(DirectX::XMVector3Length(DirectX::XMVectorSubtract(
        DirectX::XMVectorSet(DirectX::XMVectorGetX(camPos), 0.0f, DirectX::XMVectorGetZ(camPos), 0.0f),
        DirectX::XMVectorSet(transform.Position.x + CHUNK_SIZE_X / 2, 0.0f,
            transform.Position.z + CHUNK_SIZE_Z / 2, 0.0f))));
    size_t lod_id = size_t(dist / lod_dist);
    if (lod_id >= m_lod_geometries.size())
        lod_id = m_lod_geometries.size() - 1;

    if (idCamX <= idChunkX)
        DrawGeometry(m_lod_geometries[lod_id].geometries[0], device);
    if (idCamX >= idChunkX)
        DrawGeometry(m_lod_geometries[lod_id].geometries[1], device);
    DrawGeometry(m_lod_geometries[lod_id].geometries[3], device);
    if (idCamZ <= idChunkZ)
        DrawGeometry(m_lod_geometries[lod_id].geometries[4], device);
    if (idCamZ >= idChunkZ)
        DrawGeometry(m_lod_geometries[lod_id].geometries[5], device);
}
