cbuffer constants{
    float4x4 MVP;
};

struct VS_INPUT {
    uint id : POSITION0;
    float4 color : COLOR;
    float3 instPosition : POSITION1;
    uint faceDir : POSITION2;
    uint2 scale : SCALE0;
};

struct VS_OUTPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float3 normal : NORMAL;
};

VS_OUTPUT main(VS_INPUT input) {
    float3 verticies[] = {
        float3( 0.0f, 0.0f, 0.0f ), // 0
        float3( 0.0f, 1.0f, 0.0f ), // 1
        float3( 1.0f, 1.0f, 0.0f ), // 2
        float3( 1.0f, 0.0f, 0.0f ), // 3
        float3( 0.0f, 0.0f, 1.0f ), // 4
        float3( 0.0f, 1.0f, 1.0f ), // 5
        float3( 1.0f, 1.0f, 1.0f ), // 6
        float3( 1.0f, 0.0f, 1.0f ), // 7
    };

    static const float3 faceNormals[6] = {
        float3(-1, 0, 0), // left
        float3(1, 0, 0),  // right
        float3(0, -1, 0), // down
        float3(0, 1, 0),  // up
        float3(0, 0, -1), // front
        float3(0, 0, 1)   // back
    };

    uint indices[] = {
        4, 5, 0, 1, // left
        3, 2, 7, 6, // right
        3, 7, 0, 4, // down
        1, 5, 2, 6, // up
        0, 1, 3, 2, // front
        7, 6, 4, 5, // back
    };

    VS_OUTPUT output;

    uint index = indices[input.id + input.faceDir * 4];
    float3 local = verticies[index];

    float sx = input.scale.x;
    float sy = input.scale.y;
    if (sx != 1.0f && sy != 1.0f) {
        switch (input.faceDir)
        {
            case 0: /* Left  (-X): */ local.y *= sy; local.z *= sx; break;
            case 1: /* Right (+X): */ {
                local.y *= sy; local.z *= sx;
                local.x += 1.0f * sx - 1.0f;
                break;
            }
            case 3: /* Up    (+Y): */ {
                local.x *= sx; local.z *= sy;
                local.y += 1.0f * sx - 1.0f;
                break;
            }
            case 4: /* Front (-Z): */ local.x *= sx; local.y *= sy; break;
            case 5: /* Back  (+Z): */ {
                local.x *= sx; local.y *= sy;
                local.z += 1.0f * sy - 1.0f;
                break;
            }
        }
    }
    local.xyz -= 0.5f * sx;

    float4 pos = float4(local + input.instPosition, 1.0f);
    output.position = mul(pos, MVP);
    output.color = float4(0.7, 0.7, 0.7, 1.0); // Override the colors...
    output.normal = faceNormals[input.faceDir];

    return output;
}
