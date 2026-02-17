#pragma once

#include "Device.h"
#include "../Core/Camera.h"
#include "../Core/Transform.h"

#include <d3d11.h>

struct Vertex {
    /// Coordinates
    FLOAT x, y, z;
    /// Color
    FLOAT color[4];
};

class Mesh {
public:
    Mesh() : m_vertexBuffer(NULL), m_indexBuffer(NULL) {}
    ~Mesh() {
        if (m_vertexBuffer)
            m_vertexBuffer->Release();
        if (m_indexBuffer)
            m_indexBuffer->Release();
    }

    ID3D11Buffer* GetVertexBuffer() const { return m_vertexBuffer; }
    ID3D11Buffer* GetIndexBuffer() const { return m_indexBuffer; }
    ID3D11Buffer** GetVertexBufferPtr() { return &m_vertexBuffer; }
    ID3D11Buffer** GetIndexBufferPtr() { return &m_indexBuffer; }

    virtual const Vertex* GetVertices() const = 0;
    virtual UINT GetVerticesSize() const = 0;
    virtual const WORD* GetIndices() const = 0;
    virtual UINT GetIndicesSize() const = 0;

    virtual void Initialize(Device& device);
    virtual void Draw(Device& device, Camera& camera, float lod_dist);

    Transform transform;

private:
    ID3D11Buffer* m_vertexBuffer;
    ID3D11Buffer* m_indexBuffer;
};

class Square : public Mesh {
public:
    virtual const Vertex* GetVertices() const override;
    virtual UINT GetVerticesSize() const override;
    virtual const WORD* GetIndices() const override;
    virtual UINT GetIndicesSize() const override;

private:
    static const Vertex m_vertices[];
    static const WORD m_indices[];

};

class Cube : public Mesh {
public:
    virtual const Vertex* GetVertices() const override;
    virtual UINT GetVerticesSize() const override;
    virtual const WORD* GetIndices() const override;
    virtual UINT GetIndicesSize() const override;

private:
    static const Vertex m_vertices[];
    static const WORD m_indices[];
};
