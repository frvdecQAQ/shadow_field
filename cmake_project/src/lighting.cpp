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
        save_path = path.substr(0, sub_pos) + "/data/light_simple.dat";
        std::cout << save_path << std::endl;
    }
    else if(light_type == 1)
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
        save_path = path.substr(0, sub_pos) + "/data/light_" + tmp_str + ".dat";
        endIndex = tmp_str.rfind("_");
        cube_map_path = "lightings/cross/" + tmp_str.substr(0, endIndex) + "_cross.hdr";
        std::cout << save_path << std::endl;
        std::cout << cube_map_path << std::endl;

        fclose(file);
    }
    else
    {
        char light_path[202];
        fscanf(light_config, "%s", light_path);
        std::cout << "Loading area light " << light_path << std::endl;

        std::ifstream in;
        try
        {
            in.open(light_path);
        }
        catch (...)
        {
            std::cout << "area light loaded error" << std::endl;
        }
        if (!in.is_open())
        {
            std::cout << "area light not opened!" << std::endl;
            exit(0);
        }
        std::string line, attribute;

        float x, y, z, u, v, nx, ny, nz;
        char mask;
        unsigned index_v0, index_v1, index_v2;
        std::string index_v0_str, index_v1_str, index_v2_str;

        while (getline(in, line))
        {
            std::istringstream s_line(line);
            s_line >> attribute;
            if (attribute == "#")
                continue;

            if (attribute == "v")
            {
                s_line >> x >> y >> z;

                _vertices.push_back(x);
                _vertices.push_back(y);
                _vertices.push_back(z);
            }
            else if (attribute == "vt")
            {
                s_line >> u >> v;
                _texcoords.push_back(u);
                _texcoords.push_back(v);
            }
            else if (attribute == "vn")
            {
                s_line >> nx >> ny >> nz;
                _normals.push_back(nx);
                _normals.push_back(ny);
                _normals.push_back(nz);
            }
            else if (attribute == "f")
            {
                s_line >> index_v0_str >> index_v1_str >> index_v2_str;
                index_v0 = index_from_str(index_v0_str);
                index_v1 = index_from_str(index_v1_str);
                index_v2 = index_from_str(index_v2_str);

                _indices.push_back(index_v0 - 1);
                _indices.push_back(index_v1 - 1);
                _indices.push_back(index_v2 - 1);
            }
        }
        
        _cx = _cy = _cz = 0.0f;
        int vertex_size = (int)_vertices.size();
        for (int i = 0; i < vertex_size; i += 3) {
            _cx += _vertices[i];
            _cy += _vertices[i + 1];
            _cz += _vertices[i + 2];
        }
        _cx = _cx * 3 / vertex_size;
        _cy = _cy * 3 / vertex_size;
        _cz = _cz * 3 / vertex_size;

        glm::vec3 center = glm::vec3(_cx, _cy, _cz);
        _r = 0.0f;
        for (int i = 0; i < vertex_size; i += 3) {
            glm::vec3 dis_vec = glm::vec3(_vertices[0], _vertices[1], _vertices[2]) - center;
            float tmp_dis = std::sqrt(dis_vec[0] * dis_vec[0] + dis_vec[1] * dis_vec[1] + dis_vec[2] * dis_vec[2]);
            _r = std::fmax(tmp_dis, _r);
        }
        init_x = _cx;
        init_y = _cy;
        init_z = _cz;

        rotate_mat = glm::mat4(1.0f);
        rotate_mat_inv = glm::mat4(1.0f);
        shadow_field = new glm::vec3[sphereNumber*shadowSampleNumber*band*band];
        save_path = path.substr(0, sub_pos) + "/data/area_light.dat";
        std::cout << save_path << std::endl;
        in.close();
    }
    fclose(light_config);
}

unsigned Lighting::index_from_str(const std::string& str) {
    int pos = (int)(str.length());
    int len = pos;
    for (int i = 0; i < len; ++i) {
        if (str[i] == '/') {
            pos = i;
            break;
        }
    }
    return atoi(str.substr(0, pos).c_str());
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
    if(shadow_field != nullptr)delete[] shadow_field;
}

void Lighting::init()
{
    std::cout << "Lighting probe: " << save_path << std::endl;
    std::ifstream in(save_path, std::ifstream::binary);
    glm::vec3 temp;

    in.read((char *)&_band, sizeof(int));
    std::cout << "read_band! = " << _band << std::endl;
    int band2 = _band * _band;
    if(light_type == 0 || light_type == 1)
    {
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
    }
    else
    {
        int sz = sphereNumber*shadowSampleNumber;
        for(int i = 0; i < sz; ++i)
        {
            for(int j = 0; j < band2; ++j)
            {
                in.read((char*)&shadow_field[i*band2+j].r, sizeof(float));
                in.read((char*)&shadow_field[i*band2+j].g, sizeof(float));
                in.read((char*)&shadow_field[i*band2+j].b, sizeof(float));
            }
        }
        sz = shadowSampleNumber;
        glm::vec2 spherical;
        glm::vec3 cartesian;
        point_sample._samples.clear();
        for (int i = 0; i < sz; ++i) {
            in.read((char*)&spherical[0], sizeof(float));
            in.read((char*)&spherical[1], sizeof(float));
            cartesian.x = sin(spherical[0]) * cos(spherical[1]);
            cartesian.y = sin(spherical[0]) * sin(spherical[1]);
            cartesian.z = cos(spherical[0]);
            point_sample._samples.emplace_back(Sample(cartesian, spherical));
        }
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
    int band2 = _band*_band;
    if(light_type == 1 || light_type == 0)
    {
        int sqrtnum = (int)sqrt(sampleNumber);
        int band2 = _band * _band;
        Sampler stemp(sqrtnum);

        stemp.computeSH(_band);
        _coeffs.clear();
        _coeffs.resize(band2, glm::vec3(0.0f, 0.0f, 0.0f));

        int sample_sz = stemp._samples.size();
        float weight = 4.0f * M_PI / sample_sz;

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
    else
    {
        point_sample = Sampler((int)sqrt(shadowSampleNumber), false);
        Sampler shadow_sample((int)sqrt(shadowSampleNumber));
        shadow_sample.computeSH(_band);

        int point_sample_sz = point_sample._samples.size();
        int shadow_sample_sz = shadow_sample._samples.size();
        float weight = 4.0f * M_PI / shadow_sample_sz;
        int face_num = _indices.size()/3;

#pragma omp parallel for
        for (int t = 0; t < sphereNumber; ++t) 
        {
            float now_r = (rStart + t * rStep) * _r;
            for (int j = 0; j < point_sample_sz; ++j) 
            {
                Sample& stemp = point_sample._samples[j];
                glm::vec3 now_pos = glm::vec3(_cx, _cy, _cz) + now_r*stemp._cartesCoord;
                int i = t * shadowSampleNumber + j;
                for (int b = 0; b < band2; ++b)shadow_field[i*band2+b] = glm::vec3(0, 0, 0);
                for (int k = 0; k < shadow_sample_sz; ++k) 
                {
                    Ray testRay(now_pos, shadow_sample._samples[k]._cartesCoord);
                    bool visibility = false;
                    for(int k = 0; k < face_num; ++k){
                        int index_v0 = _indices[k*3];
                        int index_v1 = _indices[k*3+1];
                        int index_v2 = _indices[k*3+2];
                        Triangle tmp(glm::vec3(_vertices[index_v0*3], _vertices[index_v0*3+1], _vertices[index_v0*3+2]),
                                    glm::vec3(_vertices[index_v1*3], _vertices[index_v1*3+1], _vertices[index_v1*3+2]),
                                    glm::vec3(_vertices[index_v2*3], _vertices[index_v2*3+1], _vertices[index_v2*3+2]),
                                    -1);
                        visibility |= rayTriangle(testRay, tmp, false);
                    }
                    if(visibility)
                    {
                        Sample& tmp_sample = shadow_sample._samples[k];
                        for(int b = 0; b < band2; ++b){
                            shadow_field[i*band2+b] += tmp_sample._SHvalue[b];
                        }
                    }
                }
                for (int b = 0; b < band2; ++b)shadow_field[i*band2+b] *= weight;
            }
        }
    }
}

void Lighting::write2Diskbin()
{
    std::ofstream out(save_path, std::ofstream::binary);
    out.write((char *)&_band, sizeof(int));
    int band2 = _band * _band;
    if(light_type == 0 || light_type == 1)
    {
        for (int i = 0; i < band2; i++)
        {
            out.write((char *)&_coeffs[i].x, sizeof(float));
            out.write((char *)&_coeffs[i].y, sizeof(float));
            out.write((char *)&_coeffs[i].z, sizeof(float));
        }
    }
    else
    {
        int sz = sphereNumber*shadowSampleNumber;
        for(int i = 0; i < sz; ++i)
        {
            for(int j = 0; j < band2; ++j)
            {
                out.write((char*)&shadow_field[i*band2+j].r, sizeof(float));
                out.write((char*)&shadow_field[i*band2+j].g, sizeof(float));
                out.write((char*)&shadow_field[i*band2+j].b, sizeof(float));
            }
        }

        sz = point_sample._samples.size();
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < 2; ++j) {
                out.write((char*)&point_sample._samples[i]._sphericalCoord[j], sizeof(float));
            }
        }
    }
    out.close();
    std::cout << "Lighting probe generated." << std::endl;
}

void Lighting::querySRF(glm::vec3 p, glm::vec3* coef) {
    assert(light_type == 2);
    const int band2 = _band * _band;
    for (int i = 0; i < band2; ++i)coef[i] = glm::vec3(0, 0, 0);

    float nx, ny, nz;
    
    nx = p[0];
    ny = p[1];
    nz = p[2];
    p[0] = rotate_mat_inv[0][0] * nx + rotate_mat_inv[0][1] * ny + rotate_mat_inv[0][2] * nz + rotate_mat_inv[0][3];
    p[1] = rotate_mat_inv[1][0] * nx + rotate_mat_inv[1][1] * ny + rotate_mat_inv[1][2] * nz + rotate_mat_inv[1][3];
    p[2] = rotate_mat_inv[2][0] * nx + rotate_mat_inv[2][1] * ny + rotate_mat_inv[2][2] * nz + rotate_mat_inv[2][3];
    glm::vec3 p_r_vec = p - glm::vec3(init_x, init_y, init_z);

    float p_r_dis = std::sqrt(p_r_vec[0] * p_r_vec[0] + p_r_vec[1] * p_r_vec[1] + p_r_vec[2] * p_r_vec[2]);
    if (p_r_dis < 1e-6)return;

    float theta = std::acos(p_r_vec[2] / p_r_dis);
    float phi = std::atan2(p_r_vec[1], p_r_vec[0]);
    if (phi < 0)phi += 2 * M_PI;

    int n = std::sqrt(shadowSampleNumber);
    float tmp_cos = std::cos(theta);
    float x = (1 - tmp_cos) / 2;
    float y = phi / (2.0 * M_PI);
    nx = x * n;
    ny = y * n;
    float t = (p_r_dis / _r - rStart) / rStep;
    int nx_interp, ny_interp, t_interp;

    nx_interp = int(nx);
    ny_interp = int(ny);
    t_interp = int(t);

    int tmp_t[2], tmp_x[2], tmp_y[2];
    if (t_interp >= sphereNumber - 1) {
        tmp_t[0] = tmp_t[1] = sphereNumber - 1;
    }
    else if (t < 0) {
        tmp_t[0] = tmp_t[1] = 0;
    }
    else {
        tmp_t[0] = t_interp;
        tmp_t[1] = t_interp + 1;
    }

    if (nx_interp >= n - 1) {
        tmp_x[0] = tmp_x[1] = n - 1;
    }
    else {
        tmp_x[0] = nx_interp;
        tmp_x[1] = nx_interp + 1;
    }

    if (ny_interp >= n - 1) {
        tmp_y[0] = n - 1;
        tmp_y[1] = 0;
    }
    else {
        tmp_y[0] = ny_interp;
        tmp_y[1] = ny_interp + 1;
    }
    float sum = 0;
    //std::cout << "OOF query check point 2" << std::endl;
    for (int i = 0; i < 2; ++i) {
        float now_r = (tmp_t[i] * rStep + rStart) * _r;
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                int index = tmp_x[j] * n + tmp_y[k];
                float theta_index = point_sample._samples[index]._sphericalCoord[0];
                float phi_index = point_sample._samples[index]._sphericalCoord[1];
                float theta_minus = fabs(theta_index - theta) * n / M_PI + 1e-6;
                float phi_minus = fabs(phi_index - phi) * n / (2 * M_PI) + 1e-6;
                float r_minus = fabs(now_r - p_r_dis) / (rStep * _r) + 1e-6;
                float coef_tmp = 1.0 / (theta_minus * phi_minus * r_minus);
                int pos = tmp_t[i] * shadowSampleNumber + index;
                sum += coef_tmp;
                for (int b = 0; b < band2; ++b) {
                    coef[b] += shadow_field[pos*band2+b] * coef_tmp;
                }
            }
        }
    }
    for (int i = 0; i < band2; ++i)coef[i] /= sum;
}