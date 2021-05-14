#version 430

in vec3 Color;
in vec2 texCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;

void main()
{
    FragColor = texture(ourTexture, texCoord)*vec4(Color, 1.0);
}

