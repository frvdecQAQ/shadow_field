#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <io.h>
#include <cmath>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

// GLEW
// #define GLEW_STATIC
#include <GL/glew.h>
// GLFW
#include <GLFW/glfw3.h>
// GLM
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
// AntTweakBar
#include <AntTweakBar.h>

#include "UI.h"
#include "brdf.h"
#include "scene.h"
#include "lighting.h"
#include "diffuseObject.h"
#include "generalObject.h"
#include "resource_manager.h"
#include "multi_product.h"
#include "renderer.h"
#include "shorder.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#define FULL_SCREEN

// Window size.
int WIDTH, HEIGHT;
GLuint width = 1920, height = 1080;

// Keyboard.
bool keys[1024];

// Mouse.
GLfloat lastX = WIDTH / 2.0f, lastY = HEIGHT / 2.0f;
bool firstMouse = true;

const int LightNumber = 5;

std::string objects[] = {"buddha", "maxplanck"};
std::string gobjects[] = {"buddha", "maxplanck"};
std::string lightings[] = {"galileo", "grace", "rnl", "stpeters", "uffizi"};
std::string bands[] = {"linear", "quadratic", "cubic", "quartic"};
BRDF_TYPE BRDFs[] = {BRDF_PHONG, BRDF_WARD_ISOTROPIC, BRDF_WARD_ANISOTROPIC};

glm::vec3 albedo(0.12f, 0.12f, 0.12f);

Scene scene;
DiffuseObject diffObject;
GeneralObject genObject;
Lighting *now_light;
BRDF brdf;
Renderer renderer;

int sampleNumber = 128 * 128;
int band = 5;
int sphereNumber = 32;
int shadowSampleNumber = 48 * 48;

// Cubemap & Simple Light.
bool renderBar = true;

// Camera.
float camera_dis = 1.1f;
glm::vec3 camera_pos(0.789035f, 0.290739f, -0.143193f);
glm::vec3 last_camera_pos(0.0f, 0.0f, 1.0f);
glm::vec3 camera_dir(0.0f, 0.0f, 0.0f);
glm::vec3 camera_up(0.0f, 1.0f, 0.0f);
glm::vec3 camera_front(0.0f, 0.0f, -1.0f);

//Simple lighting.
glm::vec3 light_dir(0.0f, 0.0f, 1.0f);
glm::vec3 last_light_dir(0.0f, 0.0f, 1.0f);

// Rotation.
glm::fquat g_Rotation(0.0f, 0.0f, 0.0f, 1.0f);
glm::fquat last_Rotation(0.0f, 0.0f, 0.0f, 1.0f);
glm::fquat g_RotateStart(0.0f, 0.0f, 0.0f, 1.0f);
glm::mat4 rotateMatrix;
int g_AutoRotate = 0;
int g_RotateTime = 0;

// FPS.
double currTime;
double lastTime;
int frames = 0;
int fps;

// GLFW
GLFWwindow* window;

/*
 *  Function prototypes.
 */
// Callback functions.
int key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
int button_calback(GLFWwindow* window, int button, int action, int mods);
int mouse_callback(GLFWwindow* window, double xpos, double ypos);
int scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
// Data processing function.
void dataProcessing(int argc, char** argv);
// Initialization functions.
void dataLoading(int argc, char** argv);
void shaderLoading();
// Miscellaneous.
void calculateFPS();

void saveImage(char* filepath, GLFWwindow* w) {
	int width, height;
	glfwGetFramebufferSize(w, &width, &height);
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
    FILE* pos_file = fopen("pos.txt", "w");
    fprintf(pos_file, "%lf %lf %lf\n", camera_pos[0], camera_pos[1], camera_pos[2]);
    fprintf(pos_file, "%lf\n", camera_dis);
    fclose(pos_file);
}



int main(int argc, char** argv){
    /*
    //GPU init
    gpu_initGamma();
    //CPU init
    SH<n>::init();
    cpu_initGamma();
    float *A, *B, *C, *D, *E, *F;
    float *IA;
    cufftComplex *pool0, *pool1, *pool2;
    int num = 1024;
    cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*num);
    cudaMalloc((void**)&pool1, sizeof(cufftComplex)*N*N*num);
    cudaMalloc((void**)&pool2, sizeof(cufftComplex)*N*N*num);
    cudaMalloc(&A, sizeof(float)*n*n*num);
    cudaMalloc(&B, sizeof(float)*n*n*num);
    cudaMalloc(&C, sizeof(float)*n*n*num);
    cudaMalloc(&D, sizeof(float)*n*n*num);
    cudaMalloc(&E, sizeof(float)*n*n*num);
    cudaMalloc(&F, sizeof(float)*n*n*num);
    cufftHandle plan;
    int sizes[2] = {N,N};
	cufftPlanMany(&plan, 2, sizes, NULL, 1, N*N, NULL, 1, N*N, CUFFT_C2C, num);
    IA = new float[n*n*num];
    for(int i = 0; i < n*n*num; ++i){
        IA[i] = ((i&1)?-1.0f:1.0f);
    }
    cudaMemcpy(A, IA, num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, IA, num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C, IA, num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(D, IA, num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(E, IA, num*n*n*sizeof(float), cudaMemcpyHostToDevice);
    shprod_many(A, B, C, D, E, F, pool0, pool1, pool2, num, plan);
    cudaMemcpy(IA, F, num*n*n*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << IA[0] << std::endl;
    multi_product(A, B, C, D, E, F, num, 0);
    cudaMemcpy(IA, F, num*n*n*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << IA[0] << std::endl;
    multi_product(A, B, C, D, E, F, num, 1);
    cudaMemcpy(IA, F, num*n*n*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << IA[0] << std::endl;
    delete[] IA;

    SH<n> sh1, sh2, sh3, sh4, sh5;
    for(int l = 0; l < n; ++l){
        for(int m = -l; m <= l; ++m){
            int index = l*(l+1)+m;
            sh1.at(l, m) = ((index&1)?-1.0f:1.0f);
            sh2.at(l, m) = ((index&1)?-1.0f:1.0f);
            sh3.at(l, m) = ((index&1)?-1.0f:1.0f);
            sh4.at(l, m) = ((index&1)?-1.0f:1.0f);
            sh5.at(l, m) = ((index&1)?-1.0f:1.0f);
        }
    }

    SH<n> sh6 = precise(sh1,sh2,sh3,sh4,sh5);

    std::cout << sh6.at(0, 0) << std::endl;

    SH<n> sh7 = fs2sh(fastmul(sh2fs(sh1),sh2fs(sh2),sh2fs(sh3),sh2fs(sh4),sh2fs(sh5)));

    std::cout << sh7.at(0, 0) << std::endl;

    SH<n> sh8 = sh1*sh2*sh3*sh4*sh5;

    std::cout << sh8.at(0, 0) << std::endl;

    releaseGamma();
    return 0;*/

    dataProcessing(argc, argv);
    renderer.loadTriple(band);
    gpu_initGamma();
    SH<n>::init();
    cpu_initGamma();
    // Init GLFW.
    glfwInit();
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_AUTO_ICONIFY, GL_FALSE);
#endif
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
     glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
     glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef FULL_SCREEN
     // Create a "Windowed full screen" window in the primary monitor.
     GLFWmonitor* monitor = glfwGetPrimaryMonitor();
     const GLFWvidmode* mode = glfwGetVideoMode(monitor);
     glfwWindowHint(GLFW_RED_BITS, mode->redBits);
     glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
     glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
     glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
     window = glfwCreateWindow(mode->width, mode->height, "prt-SH", monitor, nullptr);
#else
     window = glfwCreateWindow(width, height, "prt-SH", nullptr, nullptr);
#endif
    glfwMakeContextCurrent(window);

    // Note: The required callback functions are set with AntTweakBar UI.

    // Set this to true so GLEW knows to use a modern approach to retrieving function pointers and extensions.
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers.
    glewInit();

    // Define the viewport dimensions.
    glfwGetFramebufferSize(window, &WIDTH, &HEIGHT);
    glViewport(0, 0, WIDTH, HEIGHT);

    width = WIDTH;
    height = HEIGHT;

    std::cout << "GLFW version                : " << glfwGetVersionString() << std::endl;
    std::cout << "GL_VERSION                  : " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GL_VENDOR                   : " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "GL_RENDERER                 : " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "GL_SHADING_LANGUAGE_VERSION : " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "WINDOW WIDTH                : " << WIDTH << std::endl;
    std::cout << "WINDOW HEIGHT               : " << HEIGHT << std::endl;

    // Setup some OpenGL options.
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthFunc(GL_LESS);
    // Anti-aliasing.
    glEnable(GL_MULTISAMPLE);
    glfwSwapInterval(0);

    // Do some initialization (including loading data, shaders, models, etc.)
    std::cout << "load start" << std::endl;
    dataLoading(argc, argv);
    std::cout << "load done" << std::endl;
    //scene.prepareData(30);
    //std::cout << "prepare done" << std::endl;
    //return 0;
    shaderLoading();
    renderer.Init(LightNumber);
    UIInit();
    lastTime = glfwGetTime();

    int translate_cnt = 0;
    bool translate_flag = true;

    //scene.change();

    // Loop.

    std::cout << "======================Render Start====================" << std::endl;
    double render_time = 0;
    int render_cnt = 0;
    while (!glfwWindowShouldClose(window))
    {
        double start_time = glfwGetTime();
        // Calculate FPS.
        calculateFPS();

        // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions.
        glfwPollEvents();

        // Clear the color buffer.
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Render.
        //int f;
        //scanf("%d", &f);
        //scene.change(camera_pos, camera_dir);
        bool render_again = true;
        renderer.Render(render_again);

        // Render AntTweakBar UI.
        if(renderBar) TwDraw();

        // Swap the screen buffers.
        glfwSwapBuffers(window);
        double end_time = glfwGetTime();
        std::cout << "time = " << end_time - start_time << std::endl;
        render_cnt++;
        render_time += (end_time - start_time);
        std::cout << "avg time = " << render_time/render_cnt << std::endl;

        std::cout << " ============================ " << std::endl;

    }
    // Terminate GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();
    releaseGamma();
    return 0;
}

void calculateFPS()
{
    // Return current time in seconds.
    currTime = glfwGetTime();
    frames++;

    if ((currTime - lastTime) >= 1.0f)
    {
        fps = frames / (currTime - lastTime);
        frames = 0;
        lastTime = currTime;
    }
}

void dataLoading(int argc, char** argv)
{
    std::cout << "Loading data ................ " << std::endl;

    std::string ptype(argv[1]);
    assert(ptype == "-s" || ptype == "-l");

    std::string diffGeneal(argv[2]);
    assert(diffGeneal == "-d" || diffGeneal == "-g");

    band = n;
    std::string path(argv[4]);

    assert(access((path+"/data").c_str(), 0) != -1);

    now_light = new Lighting(path+"/light_config.txt", band);
    now_light->init();

    int tmp_cnt = 0;
    for (int i = 0; i < scene.obj_num; ++i) {
        std::string obj_name = path + "/" + scene.name_list[i];
        size_t beginIndex = obj_name.rfind('/');
        size_t endIndex = obj_name.rfind('.');
        std::string save_path = path + "/data/" + obj_name.substr(beginIndex + 1, endIndex - beginIndex - 1) + "U.dat";
        Object* tmp;
        if (scene.type_list[i] == 0)tmp = new DiffuseObject();
        else tmp = new GeneralObject();
        tmp->init(obj_name, albedo);
        tmp_cnt++;
        tmp->readFDiskbin(save_path);
        scene.obj_list.push_back(tmp);
    }scene._band = band;
    scene.prepare();

    renderer.Setup(&scene, now_light);
    renderer.SetupColorBuffer(0, camera_dis * camera_pos, false);

    std::cout << "Loading Done" << std::endl;
}

void shaderLoading()
{
    std::cout << "Loading shaders ............. ";
    ResourceManager::LoadShader(_SHADER_PREFIX_"/cubemap.vert", _SHADER_PREFIX_"/cubemap.frag", "", "cubemap");
    ResourceManager::LoadShader(_SHADER_PREFIX_"/prt.vert", _SHADER_PREFIX_"/prt.frag", "", "prt");
    std::cout << "Done" << std::endl;
}

bool check_light_process(std::string path, int band)
{
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int tmp_band;
    in.read((char*)&tmp_band, sizeof(int));
    in.close();
    if (tmp_band == band)return true;
    return false;
}

bool check_obj_process(std::string path, int type, int band, int sample, int sphere, int shadow) 
{
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int size, tmp_band, tmp_sample, tmp_sphere, tmp_shadow;
    if (type == 0) {
        in.read((char*)&size, sizeof(int));
        in.read((char*)&tmp_band, sizeof(int));
        in.read((char*)&tmp_sample, sizeof(int));
        in.read((char*)&tmp_sphere, sizeof(int));
        in.read((char*)&tmp_shadow, sizeof(int));
        in.close();
        if (tmp_band != band)return false;
        if (tmp_sample != sample)return false;
        if (tmp_sphere != sphere)return false;
        if (tmp_shadow != shadow)return false;
    }
    else {
        in.read((char*)&size, sizeof(int));
        in.read((char*)&tmp_band, sizeof(int));
        in.close();
        if (tmp_band != band)return false;
    }
    return true;
}

bool check_brdf_process(std::string path, int band) {
    std::ifstream in(path, std::ifstream::binary);
    if (!in)return false;
    int tmp_band;
    in.read((char*)&tmp_band, sizeof(int));
    in.close();
    if (tmp_band != band)return false;
    return true;
}

void dataProcessing(int argc, char** argv)
{
    std::string ptype(argv[1]);
    assert(ptype == "-s" || ptype == "-l");

    std::string diffGeneal(argv[2]);
    assert(diffGeneal == "-d" || diffGeneal == "-g");

    band = n;

    std::string path(argv[4]);
    scene.init(path);


    if(access((path+"/data").c_str(), 0) == -1)mkdir((path+"/data").c_str(), 0777);

    Lighting pattern(path+"/light_config.txt", band);
    bool exist_flag = check_light_process(pattern.save_path, band);
    if (!exist_flag)
    {
        pattern.process(sampleNumber);
        pattern.write2Diskbin();
    }

    std::vector<std::string>file_name;
    for (int i = 0; i < scene.obj_num; ++i) {
        file_name.push_back(path + "/" + scene.name_list[i]);
    }
 
    int transferType = 2;
    int tmp_cnt = 0;
    for (auto obj_path : file_name) {
        std::cout << obj_path << std::endl;
        size_t beginIndex = obj_path.rfind('/');
        size_t endIndex = obj_path.rfind('.');
        std::string save_path = path + "/data/" + obj_path.substr(beginIndex + 1, endIndex - beginIndex - 1) + "U.dat";
        bool exist_flag = check_obj_process(save_path, scene.type_list[tmp_cnt],
            band, sampleNumber, sphereNumber, shadowSampleNumber);
        if (!exist_flag)
        {
            Object* tmp;
            if (scene.type_list[tmp_cnt] == 1)tmp = new GeneralObject();
            else tmp = new DiffuseObject();
            tmp->init(obj_path.c_str(), albedo);
            //std::cout << obj_path << std::endl;
            //std::cout << tmp->_indices.size() << std::endl;
            tmp->project2SH(transferType, band, sampleNumber, 1);
            tmp->write2Diskbin(save_path);
        }
        tmp_cnt++;
    }
    //fftprecomputed fs_pre;
    //fs_pre.init();
}

// Is called whenever a key is pressed/released via GLFW.
int key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);

    float cameraSpeed = 0.2;
    if (key == GLFW_KEY_W && action == GLFW_PRESS)
        camera_pos += cameraSpeed * camera_front;
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
        camera_pos -= cameraSpeed * camera_front;
    if (key == GLFW_KEY_A && action == GLFW_PRESS)
        camera_pos -= cameraSpeed * glm::normalize(glm::cross((camera_dir - camera_dis * camera_pos), camera_up));
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
        camera_pos += cameraSpeed * glm::normalize(glm::cross((camera_dir - camera_dis * camera_pos), camera_up));
    if (key == GLFW_KEY_K && action == GLFW_PRESS)
        saveImage("./result.png", window);
    if (key == GLFW_KEY_H && action == GLFW_PRESS)
        renderBar = !renderBar;

    if (key >= 0 && key < 1024)
    {
        if (action == GLFW_PRESS)
            keys[key] = true;
        else if (action == GLFW_RELEASE)
            keys[key] = false;
    }

    return TwEventKeyGLFW(key, action);
}

int mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    lastX = xpos;
    lastY = ypos;

    return TwMouseMotion((int)xpos, (int)ypos);
}

// Is called whenever MOUSE_LEFT or MOUSE_RIGHT is pressed/released via GLFW.
int button_calback(GLFWwindow* window, int button, int action, int mods)
{
    return TwEventMouseButtonGLFW(button, action);
}


int scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (yoffset > 0 && camera_dis >= 1.0f)
    {
        camera_dis -= 0.2f;
    }
    else
    {
        camera_dis += 0.2f;
    }
    return TwEventMouseWheelGLFW(yoffset);
}