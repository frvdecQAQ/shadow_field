#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "renderer.h"
#include "UI.h"
#include "resource_manager.h"
#include "sphericalHarmonics.h"
#include "brdf.h"


extern bool drawCubemap;
extern bool simpleLight;

extern std::string lightings[];
extern int lightingIndex;
extern int objectIndex;
extern int bandIndex;
extern int BRDFIndex;

// Window.
extern int WIDTH;
extern int HEIGHT;

// Camera.
extern float camera_dis;
extern glm::vec3 camera_pos;
extern glm::vec3 last_camera_pos;
extern glm::vec3 camera_dir;
extern glm::vec3 camera_up;

// Rotation.
extern int g_AutoRotate;
extern int g_RotateTime;
extern glm::fquat last_Rotation;
extern glm::mat4 rotateMatrix;

// Mesh information.
int vertices;
int faces;

Renderer::~Renderer()
{
    for(int i = 0; i < 6; ++i){
        if(cpu_data[i] != nullptr)delete[] cpu_data[i];
    }
    delete[]hdrTextures;
}

void Renderer::loadTriple(int _band) {
    band = _band;
    std::string sparse_file_path = "./processedData/triple/sparse" + std::to_string(band);
    FILE* sparse_file = fopen(sparse_file_path.c_str(), "r");
    std::cout << sparse_file_path << std::endl;
    char ch;
    int number_cnt = 0;
    int tmp_int = 0;
    double tmp_double, tmp_back;
    std::pair<int, int> tmp_pair;
    bool tmp_flag = false;
    while (true) {
        ch = fgetc(sparse_file);
        if (ch == EOF)break;
        if (ch == ',') {
            if (number_cnt == 0)dst.push_back(tmp_int);
            else if (number_cnt == 1)tmp_pair.first = tmp_int;
            else if (number_cnt == 2) {
                tmp_pair.second = tmp_int;
                src.push_back(tmp_pair);
            }
            else {
                if (tmp_flag)tmp_double = -tmp_double;
                coef.push_back(tmp_double);
            }
            number_cnt++;
            if (number_cnt == 4)number_cnt = 0;
            if (number_cnt < 3)tmp_int = 0;
            else {
                tmp_double = 0;
                tmp_back = 0;
                tmp_flag = false;
            }
        }
        if (ch != '-' && ch != '.' && (ch < '0' || ch > '9'))continue;
        if (ch == '.')tmp_back = 0.1;
        else if (ch == '-')tmp_flag = true;
        else if (number_cnt < 3)tmp_int = tmp_int * 10 + ch - '0';
        else {
            if (tmp_back != 0) {
                tmp_double += (ch - '0') * tmp_back;
                tmp_back *= 0.1;
            }
            else tmp_double = tmp_double * 10 + ch - '0';
        }
    }
    int sz = (int)(dst.size());
    fclose(sparse_file);
}

void Renderer::Init(const int lightNumber)
{
    // Initialize cubemap.
    hdrTextures = new HDRTextureCube[lightNumber];
    for (int i = 0; i < lightNumber; i++)
    {
        hdrTextures[i].Init("lightings/cross/" + lightings[i] + "_cross" + ".hdr");
    }

    // Initialize projection matrix.
    projection = glm::perspective(ZOOM, (float)WIDTH / (float)HEIGHT, NEAR_PLANE, FAR_PLANE);
}

void Renderer::Setup(Scene* scene, Lighting* light){
    _scene = scene;
    _lighting = light;

    multi_product_num = 0;
    for(int obj_id = 0; obj_id < scene->obj_num; ++obj_id){
        int vertex_num = scene->obj_list[obj_id]->_vertices.size()/3;
        multi_product_num += vertex_num;
    }
    cudaMalloc(&gpu_data[0], sizeof(float)*multi_product_num*n*n);
    cudaMalloc(&gpu_data[1], sizeof(float)*multi_product_num*n*n);
    cudaMalloc(&gpu_data[2], sizeof(float)*multi_product_num*n*n);
    cudaMalloc(&gpu_data[3], sizeof(float)*multi_product_num*n*n);
    cudaMalloc(&gpu_data[4], sizeof(float)*multi_product_num*n*n);
    cudaMalloc(&gpu_data[5], sizeof(float)*multi_product_num*n*n);
    cudaMalloc((void**)&gpu_pool0, sizeof(cufftComplex)*N*N*multi_product_num);
    cudaMalloc((void**)&gpu_pool1, sizeof(cufftComplex)*N*N*multi_product_num);
    cudaMalloc((void**)&gpu_pool2, sizeof(cufftComplex)*N*N*multi_product_num);
    for(int i = 0; i < 6; ++i)cpu_data[i] = new float[multi_product_num*n*n];

    int sizes[2] = {N,N};
	cufftPlanMany(&plan, 2, sizes, NULL, 1, N*N, NULL, 1, N*N, CUFFT_C2C, multi_product_num);
}

void Renderer::SetupColorBuffer(int type, glm::vec3 viewDir, bool diffuse)
{
    setupBuffer(type, viewDir);
}

void Renderer::our_multi_product(float* a, float* b, float* c, float* d, float*e ,float *f) {
    SH<n> sh1, sh2, sh3, sh4, sh5, sh6;
    int index;
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            sh1.at(l, m) = a[index];
            sh2.at(l, m) = b[index];
            sh3.at(l, m) = c[index];
            sh4.at(l, m) = d[index];
            sh5.at(l, m) = e[index];
        }
    }
    sh6 = fs2sh(fastmul(sh2fs(sh1),sh2fs(sh2),sh2fs(sh3),sh2fs(sh4),sh2fs(sh5)));
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            f[index]= sh6.at(l, m);
        }
    }
}

void Renderer::precise_multi_product(float *a, float *b, float *c, float *d, float *e, float *f) {
    SH<n> sh1, sh2, sh3, sh4, sh5, sh6;
    int index;
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            sh1.at(l, m) = a[index];
            sh2.at(l, m) = b[index];
            sh3.at(l, m) = c[index];
            sh4.at(l, m) = d[index];
            sh5.at(l, m) = e[index];
        }
    }
    sh6 = precise(sh1, sh2, sh3, sh4, sh5);
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            f[index]= sh6.at(l, m);
        }
    }
}

void Renderer::brute_multi_product(float *a, float *b, float *c, float *d, float *e, float *f){
    SH<n> sh1, sh2, sh3, sh4, sh5, sh6;
    int index;
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            sh1.at(l, m) = a[index];
            sh2.at(l, m) = b[index];
            sh3.at(l, m) = c[index];
            sh4.at(l, m) = d[index];
            sh5.at(l, m) = e[index];
        }
    }
    sh6 = sh1*sh2*sh3*sh4*sh5;
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            index = l*(l+1)+m;
            f[index]= sh6.at(l, m);
        }
    }
}

float Renderer::testCoef(float* coef, float theta, float phi) {
    int band = _scene->_band;
    int band2 = band * band;

    float* base = new float[band2];
    for (int l = 0; l < band; ++l) {
        for (int m = -l; m <= l; ++m) {
            int index = l * (l + 1) + m;
            base[index] = (float)SphericalH::SHvalue(theta, phi, l, m);
        }
    }
    float result = 0;
    for (int i = 0; i < band2; ++i) {
        result += coef[i] * base[i];
        //std::cout << "coef = " << coef[i] << ' ' << "base =" << base[i] << std::endl;
    }
    delete[] base;
    return result;
}

void Renderer::testMap(float* coef, const std::string& path) {
    const int tmp_n = 128;
    cv::Mat gray(tmp_n, tmp_n, CV_32FC1);
    for (int i = 0; i < tmp_n; ++i) {
        for (int j = 0; j < tmp_n; ++j) {
            float x = float(i) / float(tmp_n);
            float y = float(j) / float(tmp_n);
            float theta = std::acos(1 - 2 * x);
            float phi = y * 2 * M_PI;
            float now_value = testCoef(coef, theta, phi);
            gray.at<float>(i, j) = now_value * 255.0;
        }
    }
    cv::imwrite(path, gray);
}

void Renderer::setupBuffer(int type, glm::vec3 viewDir)
{
    double start_time = glfwGetTime();

    int band = _scene->_band;
    int band2 = band * band;
    int sz = (int)_scene->obj_list.size();
    _meshBuffer.clear();

    int vertex_number;
    int base_index = 0;
    for(int obj_id = 0; obj_id < sz; ++obj_id)
    {
        vertex_number = _scene->obj_list[obj_id]->_vertices.size()/3;
        DiffuseObject* obj_now = dynamic_cast<DiffuseObject*>(_scene->obj_list[obj_id]);
        for(int i = 0; i < vertex_number; ++i)
        {
            int offset = 3*i;
            glm::vec3 now_point = glm::vec3(obj_now->_vertices[offset], obj_now->_vertices[offset + 1],
                obj_now->_vertices[offset + 2]);
            for(int query_id = 0; query_id < sz; ++query_id)
            {
                if(query_id == obj_id)
                {
                    for(int j = 0; j < band2; ++j)
                    {
                        cpu_data[query_id][base_index+j] = obj_now->_DTransferFunc[i][j].r;
                    }
                }
                else
                {
                    _scene->obj_list[query_id]->queryOOF(now_point, cpu_data[query_id]+base_index);
                }
            }
            base_index += band2;
        }
    }

    double end_time = glfwGetTime();
    std::cout << "time 0 = " << end_time-start_time << std::endl;
    start_time = end_time;
    
    for(int i = 0; i < 5; ++i)
    {
        cudaMemcpy(gpu_data[i], cpu_data[i], sizeof(float)*multi_product_num*band2, cudaMemcpyHostToDevice);
    }
    //multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], gpu_data[5],
    //    multi_product_num, 1);
    shprod_many(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], gpu_data[5], 
            gpu_pool0, gpu_pool1, gpu_pool2, multi_product_num, plan);
    cudaMemcpy(cpu_data[5], gpu_data[5], sizeof(float)*multi_product_num*band2, cudaMemcpyDeviceToHost);
    
    /*for(int i = 0; i < multi_product_num; ++i){
        our_multi_product(cpu_data[0]+i*band2, cpu_data[1]+i*band2, cpu_data[2]+i*band2,
                          cpu_data[3]+i*band2, cpu_data[4]+i*band2, cpu_data[5]+i*band2);
        //precise_multi_product(cpu_data[0]+i*band2, cpu_data[1]+i*band2, cpu_data[2]+i*band2,
        //                    cpu_data[3]+i*band2, cpu_data[4]+i*band2, cpu_data[5]+i*band2);
        //brute_multi_product(cpu_data[0]+i*band2, cpu_data[1]+i*band2, cpu_data[2]+i*band2,
        //                    cpu_data[3]+i*band2, cpu_data[4]+i*band2, cpu_data[5]+i*band2);
    }*/

    end_time = glfwGetTime();
    std::cout << "time 1 = " << end_time-start_time << std::endl;
    start_time = end_time;

    base_index = 0;
    for (int obj_id = 0; obj_id < sz; ++obj_id) 
    {
        int vertex_number = _scene->obj_list[obj_id]->_vertices.size() / 3;
        _colorBuffer.clear();
        _colorBuffer.resize(vertex_number * 3);
        DiffuseObject* obj_now = dynamic_cast<DiffuseObject*>(_scene->obj_list[obj_id]);

        for (int i = 0; i < vertex_number; i++) 
        {
            int offset = 3 * i;

            float cr, cg, cb;
            cr = cg = cb = 0.0f;
            
            //compute shading
            for (int j = 0; j < band2; j++)
            {
                float& multi_product_result = cpu_data[5][base_index+j];
                cr += _lighting->_Vcoeffs[0](j) * multi_product_result;
                cg += _lighting->_Vcoeffs[1](j) * multi_product_result;
                cb += _lighting->_Vcoeffs[2](j) * multi_product_result;
            }

            cr *= _scene->color[obj_id].r;
            cg *= _scene->color[obj_id].g;
            cb *= _scene->color[obj_id].b;

            _colorBuffer[offset] = cr;
            _colorBuffer[offset + 1] = cg;
            _colorBuffer[offset + 2] = cb;
            
            base_index += band2;
        }
        // Generate mesh buffer.
        int facenumber = _scene->obj_list[obj_id]->_indices.size() / 3;
        for (int i = 0; i < facenumber; i++)
        {
            int offset = 3 * i;
            int index[3] = {
                _scene->obj_list[obj_id]->_indices[offset + 0],
                _scene->obj_list[obj_id]->_indices[offset + 1],
                _scene->obj_list[obj_id]->_indices[offset + 2],
            };

            for (int j = 0; j < 3; j++)
            {
                int Vindex = 3 * index[j];
                MeshVertex vertex = {
                    _scene->obj_list[obj_id]->_vertices[Vindex + 0],
                    _scene->obj_list[obj_id]->_vertices[Vindex + 1],
                    _scene->obj_list[obj_id]->_vertices[Vindex + 2],
                    _colorBuffer[Vindex + 0],
                    _colorBuffer[Vindex + 1],
                    _colorBuffer[Vindex + 2]
                };
                _meshBuffer.push_back(vertex);
            }
        }
    }

    end_time = glfwGetTime();
    std::cout << "time 2 = " << end_time-start_time << std::endl;
    start_time = end_time;

    // Set the objects we need in the rendering process (namely, VAO, VBO and EBO).
    if (!_VAO)
    {
        glGenVertexArrays(1, &_VAO);
    }
    if (!_VBO)
    {
        glGenBuffers(1, &_VBO);
    }
    glBindVertexArray(_VAO);
         
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, _meshBuffer.size() * sizeof(MeshVertex), &(_meshBuffer[0]), GL_STATIC_DRAW);

    // Position attribute.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (GLvoid*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind.
    glBindVertexArray(0);
}
void Renderer::objDraw()
{
    glBindVertexArray(_VAO);
    vertices = _scene->vertice_num;
    faces = _scene->indices_num;
    glDrawArrays(GL_TRIANGLES, 0, _meshBuffer.size());

    // Unbind.
    glBindVertexArray(0);
}

void Renderer::Render(bool render_again)
{
    // Render objects.
    glm::mat4 view = glm::lookAt(camera_dis * camera_pos, camera_dir, camera_up);
    glm::mat4 model = glm::mat4(1.0f);
    Shader shader = ResourceManager::GetShader("prt");
    shader.Use();
    shader.SetMatrix4("model", model);
    shader.SetMatrix4("view", view);
    shader.SetMatrix4("projection", projection);

    std::cout << "c_dis : " << camera_dis << std::endl;
    std::cout << "c_pos : " << camera_pos[0] << ' ' << camera_pos[1] << ' ' << camera_pos[2] << std::endl;

    if (render_again)setupBuffer(0, camera_dis*camera_pos);

    objDraw();

    //std::cout << "Render done" << std::endl;

    /*if (drawCubemap)
    {
        // Render cubemap.
        shader = ResourceManager::GetShader("cubemap");
        shader.Use();
        // Remove translation from the view matrix.
        view = glm::mat4(glm::mat3(view));
        shader.SetMatrix4("view", view);
        shader.SetMatrix4("projection", projection);
        hdrTextures[lightingIndex].Draw();
    }*/
}
