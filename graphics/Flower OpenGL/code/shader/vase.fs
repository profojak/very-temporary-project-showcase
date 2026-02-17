#version 330

//-----------------------------------------------------------------------------
/**
* \file   vase.fs
* \author Jakub Profota
* \brief  Vase fragment shader.
*
* This file contains vase fragment shader.
*/
//-----------------------------------------------------------------------------


struct Material {
    vec3 ambient;    // Ambient material component.
    vec3 diffuse;    // Diffuse material component.
    vec3 specular;   // Specular material component.
    float shininess; // Shininess of material.
};

struct Light {
    vec3 ambient;   // Ambient light component.
    vec3 diffuse;   // Diffuse light component.
    vec3 specular;  // Specular light component.
    vec3 position;  // Light position.
    vec3 direction; // Light direction.
    float cone;     // Cosine of the spotlight half angle.
    float exponent; // Distribution of the energy within the light cone.
    int on;         // Whether the light is on.
};


smooth in vec3 v_position;  // Input fragment position.
smooth in vec3 v_normal;    // Input fragment normal.
smooth in vec2 v_texcoord;  // Input fragment texture coordinate.
in float v_camera_distance; // Input fragment distance from camera.


out vec4 f_color; // Output fragment color.


uniform sampler2D texture_color;  // Color texture sampler.

uniform Material material; // Material.
uniform Light directional; // Directional light.
uniform Light spotlight;   // Spotlight light.
uniform Light point;       // Point light.


vec4 directional_light(vec3 position, vec3 normal) {
    vec3 color = vec3(0.0);

    vec3 L = normalize(directional.direction);
    vec3 R = reflect(-L, normal);
    vec3 V = normalize(-position);
    float NL = max(0.0, dot(normal, L));
    float RV = max(0.0, dot(R, V));

    color += material.ambient * directional.ambient;
    color += material.diffuse * directional.diffuse * NL;
    color += material.specular * directional.specular * pow(RV, material.shininess);

    return vec4(color, 1.0);
}


vec4 spotlight_light(vec3 position, vec3 normal) {
    vec3 color = vec3(0.0);

    vec3 L = normalize(spotlight.position - position);
    vec3 R = reflect(-L, normal);
    vec3 V = normalize(-position);
    float NL = max(0.0, dot(normal, L));
    float RV = max(0.0, dot(R, V));
    float spot = max(0.0, dot(-L, spotlight.direction));

    color += material.ambient * spotlight.ambient;
    color += material.diffuse * spotlight.diffuse * NL;
    color += material.specular * spotlight.specular * pow(RV, material.shininess);

    if (spot < spotlight.cone) {
        return vec4(0.0);
    } else {
        return vec4(color * pow(spot, spotlight.exponent), 1.0);
    }
}


float f_falloff(float r, float f, float d) {
  float denom = d / r + 1.0;
  float attenuation = 1.0 / (denom * denom);
  float t = (attenuation - f) / (1.0 - f);
  return max(t, 0.0);
}


vec4 point_light(vec3 position, vec3 normal) {
    vec3 color = vec3(0.0);

    vec3 L = normalize(point.position - position);
    vec3 R = reflect(-L, normal);
    vec3 V = normalize(-position);
    float NL = max(0.0, dot(normal, L));
    float RV = max(0.0, dot(R, V));

    float distance = length(point.position - position);
    float falloff = f_falloff(2.0, point.exponent, distance);

    color += material.ambient * point.ambient * falloff;
    color += material.diffuse * point.diffuse * NL * falloff;
    color += material.specular * point.specular * pow(RV, material.shininess) *
        falloff;

    return vec4(color, 1.0);
}


/// Entry point of fragment shader.
void main() {
    vec4 color = vec4(0.3, 0.3, 0.3, 1.0);
    color += directional_light(v_position, v_normal);
    if (spotlight.on != 0) {
        color += spotlight_light(v_position, v_normal);
    }
    color += point_light(v_position, v_normal);

    float fog = clamp(exp2(-0.25 * v_camera_distance), 0.0, 1.0);
    f_color = (1 - fog) * vec4(0.5, 0.5, 0.5, 1.0) +
        fog * color * texture(texture_color, v_texcoord);
}


//-----------------------------------------------------------------------------
