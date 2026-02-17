struct PS_INPUT {
    float4 position : SV_POSITION;
    float4 color : COLOR;
    float3 normal : NORMAL;
};

float4 main(PS_INPUT input) : SV_TARGET {
    // Phong without specular highlight
    float3 N = normalize(input.normal);
    float3 L = normalize(float3(0.5f, 1.0f, 0.3f));
    float  NdotL = saturate(dot(N, L));
    float3 diffuse = input.color.rgb * NdotL;
    float3 ambient = 0.1 * input.color.rgb;
    float3 finalColor = ambient + diffuse;
    return float4(finalColor, input.color.a);
}
