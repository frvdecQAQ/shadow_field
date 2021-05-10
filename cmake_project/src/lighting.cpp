#include <cmath>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <glm/gtx/string_cast.hpp>
#include "lighting.h"
#include "rgbe.h"
#include "sampler.h"
#include "shRotation.h"
#include "simpleLighting.h"
#include "utils.h"

extern std::string bands[];

// Constructor for preprocessing.
Lighting::Lighting(std::string path, int band):
    _path(path), _band(band){
    FILE* light_config = fopen(path.c_str(), "r");
    fscanf(light_config, "%d", &light_type);
    int sub_pos = path.rfind("/");
    if (light_type == 0)
    {
        std::cout << "Lighting probe: simple light" << std::endl;
        save_path = path.substr(0, sub_pos) + "/light_simple.dat";
        std::cout << save_path << std::endl;
    }
    else
    {
        // Loading hdr textures.
        char light_path[202];
        fscanf(light_config, "%s", light_path);
        std::cout << "Loading HDR texture: " << light_path << std::endl;

        FILE* file = fopen(light_path, "rb");
        assert(file != nullptr);
        RGBE_ReadHeader(file, &_width, &_height, NULL);
        _data = new float[3 * _width * _height];
        RGBE_ReadPixels_RLE(file, _data, _width, _height);

        std::string light_path_str = std::string(light_path);

        size_t beginIndex = light_path_str.rfind('/');
        size_t endIndex = light_path_str.rfind('.');
        std::string tmp_str = light_path_str.substr(beginIndex + 1, endIndex - beginIndex - 1);
        save_path = path.substr(0, sub_pos) + "/light_" + tmp_str + ".dat";
        endIndex = tmp_str.rfind("_");
        cube_map_path = "lightings/cross/" + tmp_str.substr(0, endIndex) + "_cross.hdr";
        std::cout << save_path << std::endl;
        std::cout << cube_map_path << std::endl;

        fclose(file);
    }
    fclose(light_config);
}

/*Lighting::Lighting(int band, Eigen::VectorXf coeffs[3])
{
    _band = band;
    int band2 = band * band;

    // RGB channel.
    for (size_t i = 0; i < 3; i++)
    {
        _Vcoeffs[i].resize(band2);
        // SH coefficients.
        for (size_t j = 0; j < band2; j++)
            _Vcoeffs[i](j) = coeffs[i](j);
    }

    for (size_t i = 0; i < band2; i++)
        _coeffs.push_back(glm::vec3(coeffs[0](i), coeffs[1](i), coeffs[2](i)));
}*/

Lighting::~Lighting()
{
    if(_data != nullptr)delete[] _data;
}

void Lighting::init()
{
    std::cout << "Lighting probe: " << save_path << std::endl;
    std::ifstream in(save_path, std::ifstream::binary);
    glm::vec3 temp;

    in.read((char *)&_band, sizeof(int));
    std::cout << "read_band! = " << _band << std::endl;
    int band2 = _band * _band;
    _coeffs.clear();

    for (size_t i = 0; i < 3; i++)
    {
        _Vcoeffs[i].resize(band2);
        _Vcoeffs[i].setZero();
    }

    for (size_t i = 0; i < band2; i++)
    {
        in.read((char *)&temp.x, sizeof(float));
        in.read((char *)&temp.y, sizeof(float));
        in.read((char *)&temp.z, sizeof(float));

        _Vcoeffs[0](i) = temp.x;
        _Vcoeffs[1](i) = temp.y;
        _Vcoeffs[2](i) = temp.z;

        _coeffs.push_back(temp);
    }

    in.close();
}

// Return the light color.
// For more information about the light probe images we use:
// http://www.pauldebevec.com/Probes/
glm::vec3 Lighting::probeColor(glm::vec3 dir)
{
    dir = glm::normalize(dir);
    float d = sqrt(dir.x * dir.x + dir.y * dir.y);

    float r;
    if (fabs(d) <= M_ZERO)
    {
        r = 0.0f;
    }
    else
    {
        r = (1.0f / (2.0f * M_PI)) * acos(dir.z) / d;
    }

    glm::vec2 texCoord;
    texCoord.x = 0.5f + dir.x * r;
    texCoord.y = 0.5f + dir.y * r;

    glm::ivec2 pixelCoord;
    pixelCoord.x = (int)(_width * texCoord.x);
    pixelCoord.y = (int)(_height * (1.0f - texCoord.y));

    int index = pixelCoord.y * _width + pixelCoord.x;

    int offset = 3 * index;

    return glm::vec3(_data[offset], _data[offset + 1], _data[offset + 2]);
}

// Compute incident lighting at one or more sample points near object in terms of the SH basis.
void Lighting::process(int sampleNumber)
{
    int sqrtnum = (int)sqrt(sampleNumber);
    int band2 = _band * _band;
    // @NOTE: this weight comes from the integral of solid angle ds, referred to section 6.2 in this paper.
    Sampler stemp(sqrtnum);

    stemp.computeSH(_band);
    _coeffs.clear();
    _coeffs.resize(band2, glm::vec3(0.0f, 0.0f, 0.0f));

    int sample_sz = stemp._samples.size();
    float weight = 4.0f * M_PI / sample_sz;

    // For one channel: sampleNumber-dimension vector -> band2-dimension vector
    for (int i = 0; i < sample_sz; i++)
    {
        glm::vec3 dir = stemp._samples[i]._cartesCoord;
        for (int j = 0; j < band2; j++)
        {
            float SHvalue = stemp._samples[i]._SHvalue[j];
            if (light_type == 1)
            {
                glm::vec3 color = probeColor(dir);
                _coeffs[j] += color * SHvalue;
            }
            else
            {
                _coeffs[j] += SHvalue * Simplelight(stemp._samples[i]._sphericalCoord[0],
                                                    stemp._samples[i]._sphericalCoord[1]);
            }
        }
    }

    for (int i = 0; i < band2; ++i)
    {
        _coeffs[i] = _coeffs[i] * weight;
    }
}

void Lighting::write2Diskbin()
{
    std::ofstream out(save_path, std::ofstream::binary);
    out.write((char *)&_band, sizeof(int));
    int band2 = _band * _band;
    for (int i = 0; i < band2; i++)
    {
        out.write((char *)&_coeffs[i].x, sizeof(float));
        out.write((char *)&_coeffs[i].y, sizeof(float));
        out.write((char *)&_coeffs[i].z, sizeof(float));
    }
    out.close();
    std::cout << "Lighting probe generated." << std::endl;
}
