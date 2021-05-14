#include "object.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

Object::~Object(){
    if(light_coef != nullptr)delete[] light_coef;
    if(shadow_all != nullptr)delete[] shadow_all;
    if(texture_uv != nullptr)delete[] texture_uv;
}

void Object::normVertices(glm::vec3 scale)
{
    int size = _vertices.size();
    float scaleX = std::max(fabs(_vmaxX), fabs(_vminX));
    float scaleY = std::max(fabs(_vmaxY), fabs(_vminY));
    float scaleZ = std::max(fabs(_vmaxZ), fabs(_vminZ));

    float weight = 1.0f / Trimax(scaleX, scaleY, scaleZ);

    std::cout << weight << std::endl;
    int f;
    scanf("%d", &f);

    for (int i = 0; i < size; ++i)
    {
        _vertices[i] *= weight;
        _vertices[i] *= scale[i % 3];
    }
}

unsigned Object::index_from_str(const std::string& str) {
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

unsigned Object::index_from_str_back(const std::string& str) {
    int pos = (int)(str.length());
    int len = pos;
    for (int i = 0; i < len; ++i) {
        if (str[i] == '/') {
            pos = i;
            break;
        }
    }
    int p2 = str.rfind("/");
    return atoi(str.substr(pos+1, p2-pos-1).c_str());
}

void Object::init(std::string path, glm::vec3 albedo, bool texture)
{
    _modelname = path;
    _albedo = albedo;

    std::ifstream in;
    try
    {
        in.open(path);
    }
    catch (...)
    {
        std::cout << "Scene loaded error" << std::endl;
    }
    if (!in.is_open())
    {
        std::cout << "Obj not opened!" << std::endl;
        exit(0);
    }
    std::string line, attribute;

    float x, y, z, u, v, nx, ny, nz;
    char mask;
    unsigned index_v0, index_v1, index_v2;
    std::string index_v0_str, index_v1_str, index_v2_str;
    //unsigned index_n0, index_n1, index_n2;
    //unsigned index_t0, index_t1, index_t2;

    _vmaxX = _vmaxY = _vmaxZ = -FLT_MAX;
    _vminX = _vminY = _vminZ = FLT_MAX;

    std::cout << "here" << std::endl;

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

            f_str.push_back(index_v0_str);
            f_str.push_back(index_v1_str);
            f_str.push_back(index_v2_str);
        }
    }
    in.close();
    if(!_texcoords.empty()){
        is_texture = true;
        glGenTextures(1, &texture_map);
        glBindTexture(GL_TEXTURE_2D, texture_map);  
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        int end = path.rfind(".");
        const std::string texture_path = path.substr(0, end)+".jpg";
        cv::Mat texture_img = cv::imread(texture_path);
        width = texture_img.rows;
        height = texture_img.cols;
        nrChannels = (int)(texture_img.channels());
        GLubyte* pixels = new GLubyte[width*height*3];
        memcpy(pixels, texture_img.data, width*height*3*sizeof(char));
        if (pixels)
        {
            std::cout << "cp0" << std::endl;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
            //glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
            exit(0);
        }
        free(pixels);
        texture_uv = new float[_vertices.size()/3*2];
        for(int i = 0; i < f_str.size(); ++i){
            int v_num = index_from_str(f_str[i])-1;;
            int t_num = index_from_str_back(f_str[i])-1;
            texture_uv[v_num*2] = _texcoords[t_num*2];
            texture_uv[v_num*2+1] = _texcoords[t_num*2+1];
            std::cout << "t " << v_num << ' ' << t_num << std::endl;
        }
        std::cout << "texture done" << std::endl;
    }
    //std::cout << "???" << std::endl;
    //normVertices(scale);

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

    light_coef = new float[vertex_size*n*n];

    std::cout << "init done" << std::endl;
}

void Object::queryOOF(glm::vec3 p, float* coef, bool debug) {
    //std::cout << "OOF query check point 0" << std::endl;
    //std::cout << p[0] << ' ' << p[1] << ' ' << p[2] << std::endl;
    const int band2 = _band * _band;
    for (int i = 0; i < band2; ++i)coef[i] = 0;

    float nx, ny, nz;
    
    nx = p[0];
    ny = p[1];
    nz = p[2];
    p[0] = rotate_mat_inv[0][0] * nx + rotate_mat_inv[0][1] * ny + rotate_mat_inv[0][2] * nz + rotate_mat_inv[0][3];
    p[1] = rotate_mat_inv[1][0] * nx + rotate_mat_inv[1][1] * ny + rotate_mat_inv[1][2] * nz + rotate_mat_inv[1][3];
    p[2] = rotate_mat_inv[2][0] * nx + rotate_mat_inv[2][1] * ny + rotate_mat_inv[2][2] * nz + rotate_mat_inv[2][3];
    glm::vec3 p_r_vec = p - glm::vec3(init_x, init_y, init_z);

    //glm::vec3 p_r_vec = p-glm::vec3(_cx, _cy, _cz);
    float p_r_dis = std::sqrt(p_r_vec[0] * p_r_vec[0] + p_r_vec[1] * p_r_vec[1] + p_r_vec[2] * p_r_vec[2]);
    if (p_r_dis < 1e-6)return;

    float theta = std::acos(p_r_vec[2] / p_r_dis);
    float phi = std::atan2(p_r_vec[1], p_r_vec[0]);
    if (phi < 0)phi += 2 * M_PI;

    //std::cout << "OOF query check point 1" << std::endl;

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
                    coef[b] += _ShadowField[pos][b] * coef_tmp;
                }
            }
        }
    }

    //std::cout << "OOF query check point 3" << std::endl;
    for (int i = 0; i < band2; ++i)coef[i] /= sum;
}

bool Object::intersectTest(Ray& ray, int facenumber)
{
    bool result = false;

    // Naive approach O(n).
    for (int i = 0; i < facenumber; i++)
    {
        int offset = 3 * i;
        int index[3];
        index[0] = _indices[offset + 0];
        index[1] = _indices[offset + 1];
        index[2] = _indices[offset + 2];

        glm::vec3 p[3];
        for (int j = 0; j < 3; j++)
        {
            int Vindex = 3 * index[j];
            p[j] = glm::vec3(_vertices[Vindex], _vertices[Vindex + 1], _vertices[Vindex + 2]);
        }

        Triangle Ttemp(p[0], p[1], p[2], i);

        if (rayTriangle(ray, Ttemp))
        {
            result = true;
            break;
        }
    }

    return result;
}
