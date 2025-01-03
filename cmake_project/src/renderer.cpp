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
    /*hdrTextures = new HDRTextureCube[lightNumber];
    for (int i = 0; i < lightNumber; i++)
    {
        hdrTextures[i].Init("lightings/cross/" + lightings[i] + "_cross" + ".hdr");
    }*/

    // Initialize projection matrix.
    projection = glm::perspective(ZOOM, (float)WIDTH / (float)HEIGHT, NEAR_PLANE, FAR_PLANE);
}

void Renderer::Setup(Scene* scene, Lighting* light){
    _scene = scene;
    _lighting = light;

    if(light->light_type == 1)
    {
        hdrTextures.Init(light->cube_map_path);
        std::cout << light->cube_map_path;
    }

    multi_product_num = 0;
    for(int obj_id = 0; obj_id < scene->obj_num; ++obj_id){
        int vertex_num = scene->obj_list[obj_id]->_vertices.size()/3;
        multi_product_num += vertex_num;
    }
    if(multi_product_num%batch_size != 0){
        multi_product_num = (multi_product_num/batch_size+1)*batch_size;
    }
    cudaMalloc(&gpu_data[0], sizeof(float)*batch_size*n*n);
    cudaMalloc(&gpu_data[1], sizeof(float)*batch_size*n*n);
    cudaMalloc(&gpu_data[2], sizeof(float)*batch_size*n*n);

    if(scene->obj_num > 2)cudaMalloc(&gpu_data[3], sizeof(float)*batch_size*n*n);
    if(scene->obj_num > 3)cudaMalloc(&gpu_data[4], sizeof(float)*batch_size*n*n);
    if(scene->obj_num > 4)cudaMalloc(&gpu_data[5], sizeof(float)*batch_size*n*n);

    for(int i = 0; i < scene->obj_num+1; ++i){
        cpu_data[i] = new float[multi_product_num*n*n];
        for(int j = 0; j < multi_product_num*n*n; ++j)cpu_data[i][j] = 0.0f;
    }

    /*if(approx)
    {
        cudaMalloc((void**)&gpu_pool0, sizeof(cufftComplex)*N2*N2*batch_size);
        cudaMalloc((void**)&gpu_pool1, sizeof(cufftComplex)*N2*N2*batch_size);
        cudaMalloc((void**)&gpu_pool2, sizeof(cufftComplex)*N2*N2*batch_size);

        int sizes[2] = {N2,N2};
        cufftPlanMany(&plan, 2, sizes, NULL, 1, N2*N2, NULL, 1, N2*N2, CUFFT_C2C, batch_size);
    }
    else
    {*/
        if(scene->obj_num == 5)
        {
            cudaMalloc((void**)&gpu_pool0, sizeof(cufftComplex)*N5*N5*batch_size);
            cudaMalloc((void**)&gpu_pool1, sizeof(cufftComplex)*N5*N5*batch_size);
            cudaMalloc((void**)&gpu_pool2, sizeof(cufftComplex)*N5*N5*batch_size);

            int sizes[2] = {N5,N5};
            cufftPlanMany(&plan, 2, sizes, NULL, 1, N5*N5, NULL, 1, N5*N5, CUFFT_C2C, batch_size);
        }
        else if(scene->obj_num == 4)
        {
            cudaMalloc((void**)&gpu_pool0, sizeof(cufftComplex)*N4*N4*batch_size);
            cudaMalloc((void**)&gpu_pool1, sizeof(cufftComplex)*N4*N4*batch_size);
            cudaMalloc((void**)&gpu_pool2, sizeof(cufftComplex)*N4*N4*batch_size);

            int sizes[2] = {N4,N4};
            cufftPlanMany(&plan, 2, sizes, NULL, 1, N4*N4, NULL, 1, N4*N4, CUFFT_C2C, batch_size);
        }
        else if(scene->obj_num == 3){
            cudaMalloc((void**)&gpu_pool0, sizeof(cufftComplex)*N3*N3*batch_size);
            cudaMalloc((void**)&gpu_pool1, sizeof(cufftComplex)*N3*N3*batch_size);
            cudaMalloc((void**)&gpu_pool2, sizeof(cufftComplex)*N3*N3*batch_size);

            int sizes[2] = {N3,N3};
            cufftPlanMany(&plan, 2, sizes, NULL, 1, N3*N3, NULL, 1, N3*N3, CUFFT_C2C, batch_size);
        }
    //}
}

void Renderer::SetupColorBuffer(int type, glm::vec3 viewDir, bool diffuse)
{
    glm::mat4 view = glm::lookAt(camera_dis * camera_pos, camera_dir, camera_up);
    glm::mat4 model = glm::mat4(1.0f);
    setupBuffer(type, viewDir, model, view, projection);
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

void Renderer::setupBuffer(int type, glm::vec3 viewDir,
glm::mat4& model, glm::mat4& view, glm::mat4& projection)
{
    double start_time = glfwGetTime();

    int band = _scene->_band;
    int band2 = band * band;
    int sz = (int)_scene->obj_list.size();

    int vertex_number;
    int base_index = 0;
    //FILE* ft = fopen("sh.txt", "w");
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
                /*for(int j = 0; j < band2; ++j){
                    if(query_id == obj_id)cpu_data[query_id][base_index+j] = obj_now->_DTransferFunc[i][j].r;
                    else cpu_data[query_id][base_index+j] = obj_now->shadow_all[i*sz*band2+query_id*band2+j];
                }*/
                /*if(obj_id == 2 && i == 372)
                {
                    for(int j = 0; j < band2; ++j)fprintf(ft, "%f\n", cpu_data[query_id][base_index+j]);
                }*/
            }
            /*for(int query_id = sz; query_id < 5; ++query_id)
            {
                cpu_data[query_id][base_index] = 2.0f*sqrt(M_PI);
                for(int j = 1; j < band2; ++j)cpu_data[query_id][base_index+j] = 0.0f;
            }*/
            base_index += band2;
        }
    }

    double end_time = glfwGetTime();
    std::cout << "time 0 = " << end_time-start_time << std::endl;
    start_time = end_time;

    int loop_max = multi_product_num/batch_size;
    for(int i = 0; i < loop_max; ++i)
    {
        if(!approx)
        {
            for(int j = 0; j < _scene->obj_num; ++j)
            {
                cudaMemcpy(gpu_data[j], cpu_data[j]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            }
            if(_scene->obj_num == 5)
            {
                //multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], gpu_data[5],
                //   batch_size, 0);
                shprod_many(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], gpu_data[5], 
                        gpu_pool0, gpu_pool1, gpu_pool2, batch_size, plan);
            }
            else if(_scene->obj_num == 4)
            {
                //multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], 
                //    batch_size, 0);
                shprod_many(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4],
                    gpu_pool0, gpu_pool1, gpu_pool2, batch_size, plan);
            }
            else if(_scene->obj_num == 3)
            {
                //multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3],  
                //    batch_size, 1);
                shprod_many(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3],
                   gpu_pool0, gpu_pool1, gpu_pool2, batch_size, plan);
            }
            else if(_scene->obj_num == 1)
            {
                cudaMemcpy(gpu_data[1], cpu_data[0]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            }
            cudaMemcpy(cpu_data[_scene->obj_num]+i*batch_size*band2, gpu_data[_scene->obj_num], 
                    sizeof(float)*batch_size*band2, cudaMemcpyDeviceToHost);
        }
        else
        {
            for(int j = 0; j < _scene->obj_num; ++j)
            {
                cudaMemcpy(gpu_data[j], cpu_data[j]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            }
            if(_scene->obj_num == 5)
            {
                multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], gpu_data[5],
                   batch_size, 0);
            }
            else if(_scene->obj_num == 4)
            {
                multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3], gpu_data[4], 
                    batch_size, 0);
            }
            else if(_scene->obj_num == 3)
            {
                multi_product(gpu_data[0], gpu_data[1], gpu_data[2], gpu_data[3],  
                    batch_size, 0);
            }
            else if(_scene->obj_num == 1)
            {
                cudaMemcpy(gpu_data[1], cpu_data[0]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            }
            cudaMemcpy(cpu_data[_scene->obj_num]+i*batch_size*band2, gpu_data[_scene->obj_num], 
                    sizeof(float)*batch_size*band2, cudaMemcpyDeviceToHost);
            /*cudaMemcpy(gpu_data[0], cpu_data[0]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_data[1], cpu_data[1]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
            shprod_many(gpu_data[0], gpu_data[1], gpu_data[2], gpu_pool0, gpu_pool1, gpu_pool2,
                    batch_size, plan);
            for(int j = 2; j < _scene->obj_num; ++j)
            {
                cudaMemcpy(gpu_data[0], cpu_data[j]+i*batch_size*band2, sizeof(float)*batch_size*band2, cudaMemcpyHostToDevice);
                shprod_many(gpu_data[0], gpu_data[j], gpu_data[j+1], gpu_pool0, gpu_pool1, gpu_pool2,
                    batch_size, plan);
            }
            cudaMemcpy(cpu_data[_scene->obj_num]+i*batch_size*band2, gpu_data[_scene->obj_num], 
                    sizeof(float)*batch_size*band2, cudaMemcpyDeviceToHost);*/

        }
        /*for(int j = 0; j < batch_size; ++j)
        {
            int offset = i*batch_size*band2+j*band2;
            our_multi_product(cpu_data[0]+offset, cpu_data[1]+offset, cpu_data[2]+offset,
                            cpu_data[3]+offset, cpu_data[4]+offset, cpu_data[5]+offset);
        }*/
    }
    
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
    glm::vec3 light_coef[band2];
    for (int obj_id = 0; obj_id < sz; ++obj_id) 
    {
        int vertex_number = _scene->obj_list[obj_id]->_vertices.size() / 3;
        _colorBuffer.clear();
        _colorBuffer.resize(vertex_number * 3);
        DiffuseObject* obj_now = dynamic_cast<DiffuseObject*>(_scene->obj_list[obj_id]);

        std::cout << "obj_id = " << obj_id << std::endl;

        for (int i = 0; i < vertex_number; i++) 
        {
            int offset = 3 * i;

            float cr, cg, cb;
            cr = cg = cb = 0.0f;

            if(_lighting->light_type == 2)
            {
                glm::vec3 now_point = glm::vec3(obj_now->_vertices[offset], 
                                                obj_now->_vertices[offset + 1],
                                                obj_now->_vertices[offset + 2]);
                _lighting->querySRF(now_point, light_coef);
            }
            
            //compute shading
            for (int j = 0; j < band2; j++)
            {
                float& multi_product_result = cpu_data[sz][base_index+j];
                /*if(obj_id == 2 && i == 372)
                {
                    fprintf(ft, "%f\n", multi_product_result);
                }*/
                if(_lighting->light_type == 0 || _lighting->light_type == 1)
                {
                    //cr += _lighting->_Vcoeffs[0](j) * multi_product_result;
                    //cg += _lighting->_Vcoeffs[1](j) * multi_product_result;
                    //cb += _lighting->_Vcoeffs[2](j) * multi_product_result;
                    cr += obj_now->light_coef[i*band2*3+j]*multi_product_result;
                    cg += obj_now->light_coef[i*band2*3+band2+j]*multi_product_result;
                    cb += obj_now->light_coef[i*band2*3+band2*2+j]*multi_product_result;
                }
                else
                {
                    cr += light_coef[j].r * multi_product_result;
                    cg += light_coef[j].g * multi_product_result;
                    cb += light_coef[j].b * multi_product_result;
                }
            }
            //std::cout << cr << ' ' << cg << ' ' << cb << std::endl;
            cr *= _scene->color[obj_id].r;
            cg *= _scene->color[obj_id].g;
            cb *= _scene->color[obj_id].b;
            //std::cout << cr << ' ' << cg << ' ' << cb << std::endl;

            //cr = pow(cr, 1.0/2.2);
            //cg = pow(cg, 1.0/2.2);
            //cb = pow(cb, 1.0/2.2);
            //cr = pow(cr, 1.0f/2.2f);
            //cg = pow(cg, 1.0f/2.2f);
            //cb = pow(cb, 1.0f/2.2f);

            _colorBuffer[offset] = cr;
            _colorBuffer[offset + 1] = cg;
            _colorBuffer[offset + 2] = cb;
            
            base_index += band2;
        }
        // Generate mesh buffer.
        int facenumber = obj_now->_indices.size() / 3;
        _meshBuffer.clear();
        for (int i = 0; i < facenumber; i++)
        {
            int offset = 3 * i;
            int index[3] = {
                obj_now->_indices[offset + 0],
                obj_now->_indices[offset + 1],
                obj_now->_indices[offset + 2],
            };

            for (int j = 0; j < 3; j++)
            {
                int Vindex = 3 * index[j];
                MeshVertex vertex = {
                    obj_now->_vertices[Vindex + 0],
                    obj_now->_vertices[Vindex + 1],
                    obj_now->_vertices[Vindex + 2],
                    _colorBuffer[Vindex + 0],
                    _colorBuffer[Vindex + 1],
                    _colorBuffer[Vindex + 2],
                    (obj_now->is_texture? obj_now->texture_uv[index[j]<<1]:0),
                    (obj_now->is_texture? obj_now->texture_uv[(index[j]<<1)|1]:0)
                };
                _meshBuffer.push_back(vertex);
            }
        }
        Shader shader;
        if(obj_now->is_texture)shader = ResourceManager::GetShader("texture");
        else shader = ResourceManager::GetShader("prt");
        shader.Use();
        shader.SetMatrix4("model", model);
        shader.SetMatrix4("view", view);
        shader.SetMatrix4("projection", projection);
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

        if(obj_now->is_texture)
        {
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (GLvoid*)(6*sizeof(float)));
            glEnableVertexAttribArray(2);
            glBindTexture(GL_TEXTURE_2D, obj_now->texture_map);
        }

        // Unbind.
        //glBindVertexArray(0);

        //glBindVertexArray(_VAO);
        //vertices = _scene->vertice_num;
        //faces = _scene->indices_num;
        glDrawArrays(GL_TRIANGLES, 0, _meshBuffer.size());

        // Unbind.
        glBindVertexArray(0);
    }

    end_time = glfwGetTime();
    std::cout << "time 2 = " << end_time-start_time << std::endl;
    start_time = end_time;

    /*if(_lighting->light_type == 2)
    {
        int face_num = _lighting->_indices.size()/3;
        for(int k = 0; k < face_num; ++k)
        {
           int index[3] = {_lighting->_indices[k*3],
                           _lighting->_indices[k*3+1],
                           _lighting->_indices[k*3+2]};
           for (int j = 0; j < 3; j++)
           {
               int Vindex = 3 * index[j];
               MeshVertex vertex = {
                   _lighting->_vertices[Vindex + 0],
                   _lighting->_vertices[Vindex + 1],
                   _lighting->_vertices[Vindex + 2],
                   1,
                   1,
                   1};
               _meshBuffer.push_back(vertex);
           }
        }
    }*/

    // Set the objects we need in the rendering process (namely, VAO, VBO and EBO).
    /*if (!_VAO)
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
    glBindVertexArray(0);*/
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

void Renderer::Render(int render_cnt)
{
    /*glm::mat4 rotateMatrix = glm::mat4(1.0f);
    if((render_cnt/20)&1)rotateMatrix[2][3] = 0.03f;
    else rotateMatrix[2][3] = -0.03f;
    _lighting->rotate(rotateMatrix);*/
    // Render objects.
    glm::mat4 view = glm::lookAt(camera_dis * camera_pos, camera_dir, camera_up);
    glm::mat4 model = glm::mat4(1.0f);
    /*Shader shader = ResourceManager::GetShader("prt");
    shader.Use();
    shader.SetMatrix4("model", model);
    shader.SetMatrix4("view", view);
    shader.SetMatrix4("projection", projection);*/

    std::cout << "c_dis : " << camera_dis << std::endl;
    std::cout << "c_pos : " << camera_pos[0] << ' ' << camera_pos[1] << ' ' << camera_pos[2] << std::endl;

    setupBuffer(0, camera_dis*camera_pos, model, view, projection);

    //objDraw();

    if (_lighting->light_type == 1)
    {
        // Render cubemap.
        Shader shader = ResourceManager::GetShader("cubemap");
        shader.Use();
        // Remove translation from the view matrix.
        view = glm::mat4(glm::mat3(view));
        shader.SetMatrix4("view", view);
        shader.SetMatrix4("projection", projection);
        hdrTextures.Draw();
    }
}
