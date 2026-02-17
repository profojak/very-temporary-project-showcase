#include "Primitives.h"
#include "../Core/Camera.h"
#include "../Utilities/Logger.h"
#include "../Utilities/ErrorHandler.h"

void Mesh::Initialize(Device& device) {
    const Vertex* m_vertices = GetVertices();
    const WORD* m_indices = GetIndices();
    const UINT sizeofVertices = GetVerticesSize();
    const UINT sizeofIndices = GetIndicesSize();

    // Create vertex buffer
    // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-createbuffer
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = sizeofVertices;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    HRESULT hr = device.GetDevice()->CreateBuffer(&bd, NULL, &m_vertexBuffer);
    std::stringstream ss;
    ss << std::hex << static_cast<unsigned long>(hr);
    ThrowIfFailed(hr, "Failed to create mesh vertex buffer! Error code: ", ss.str());

    LOG_DEBUG("Created mesh vertex buffer.");

    // Copy vertices to vertex buffer
    D3D11_MAPPED_SUBRESOURCE ms;
    ThrowIfFailed(device.GetContext()->Map(m_vertexBuffer, NULL, D3D11_MAP_WRITE_DISCARD, NULL, &ms),
        "Failed to map mesh vertex buffer!");
    memcpy(ms.pData, m_vertices, sizeofVertices);
    device.GetContext()->Unmap(m_vertexBuffer, NULL);

    LOG_DEBUG("Copied mesh vertices to vertex buffer.");

    // Create index buffer
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeofIndices;
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA indexData{};
    indexData.pSysMem = m_indices;
    ThrowIfFailed(device.GetDevice()->CreateBuffer(&bd, &indexData, &m_indexBuffer),
        "Failed to create mesh index buffer!");

    LOG_DEBUG("Created mesh index buffer.");
}

void Mesh::Draw(Device& device, Camera& camera, float lod_dist) {
    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    UINT numIndices = GetIndicesSize() / sizeof(WORD);
    device.GetContext()->IASetVertexBuffers(0, 1, &m_vertexBuffer, &stride, &offset);
    device.GetContext()->IASetIndexBuffer(m_indexBuffer, DXGI_FORMAT_R16_UINT, 0);
    device.GetContext()->DrawIndexed(numIndices, 0, 0);
}

#pragma region Square_implemenation
const Vertex Square::m_vertices[] = {
    { -0.5f, -0.5f, 0.5f, { 1.0f, 0.0f, 0.0f, 1.0f }},
    { -0.5f,  0.5f, 0.5f, { 0.0f, 1.0f, 0.0f, 1.0f }},
    {  0.5f,  0.5f, 0.5f, { 0.0f, 0.0f, 1.0f, 1.0f }},
    {  0.5f, -0.5f, 0.5f, { 1.0f, 1.0f, 1.0f, 1.0f }},
};

const WORD Square::m_indices[] = {
    0, 1, 2,
    0, 2, 3,
};

const Vertex* Square::GetVertices() const { return m_vertices; }
UINT Square::GetVerticesSize() const { return sizeof(m_vertices); }
const WORD* Square::GetIndices() const { return m_indices; }
UINT Square::GetIndicesSize() const { return sizeof(m_indices); }
#pragma endregion // Square_implemenation

#pragma region Cube_implemenation
const Vertex Cube::m_vertices[] = {
    { -1.0f, -1.0f, -1.0f, { 1.0f, 0.0f, 0.0f, 1.0f }},
    { -1.0f,  1.0f, -1.0f, { 0.0f, 1.0f, 0.0f, 1.0f }},
    {  1.0f,  1.0f, -1.0f, { 0.0f, 0.0f, 1.0f, 1.0f }},
    {  1.0f, -1.0f, -1.0f, { 1.0f, 1.0f, 0.0f, 1.0f }},
    { -1.0f, -1.0f,  1.0f, { 0.0f, 1.0f, 1.0f, 1.0f }},
    { -1.0f,  1.0f,  1.0f, { 1.0f, 0.0f, 1.0f, 1.0f }},
    {  1.0f,  1.0f,  1.0f, { 1.0f, 1.0f, 1.0f, 1.0f }},
    {  1.0f, -1.0f,  1.0f, { 0.0f, 0.0f, 0.0f, 1.0f }},
};

const WORD Cube::m_indices[] = {
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    4, 5, 1, 4, 1, 0,
    3, 2, 6, 3, 6, 7,
    1, 5, 6, 1, 6, 2,
    4, 0, 3, 4, 3, 7
};

const Vertex* Cube::GetVertices() const { return m_vertices; }
UINT Cube::GetVerticesSize() const { return sizeof(m_vertices); }
const WORD* Cube::GetIndices() const { return m_indices; }
UINT Cube::GetIndicesSize() const { return sizeof(m_indices); }
#pragma endregion // Cube_implemenation
