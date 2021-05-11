#include "multi_product.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel, performs conventional O(n^5) multiplicatoin of SH vectors
// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...


__constant__ TensorEntry* deviceSparseGamma;
__constant__ int deviceSparseGammaSize;
__constant__ TensorEntry* deviceSparseGamma112;
__constant__ int deviceSparseGamma112Size;
__constant__ TensorEntry* deviceSparseGamma213;
__constant__ int deviceSparseGamma213Size;
__constant__ TensorEntry* deviceSparseGamma314;
__constant__ int deviceSparseGamma314Size;
__constant__ TensorEntry* deviceSparseGamma411;
__constant__ int deviceSparseGamma411Size;
// on CPU
std::vector<TensorEntry> SparseGamma3;

template <typename Out>
void gpu_split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> gpu_split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    gpu_split(s, delim, std::back_inserter(elems));
    return elems;
}

static std::vector<TensorEntry> readGamma(int n)
{
    int filen = -1;
    for (int a: gammalist)
        if (a>=n) {
            filen = a;
            break;
        }
    std::vector<TensorEntry> sparsegamma;
    std::ifstream sparsefile("./gamma_bin/" + std::to_string(filen) + ".bin");
    struct {int a,b,c;float k;} t;
    while (sparsefile.read((char*)(&t), 16)) {
        if (t.a<n*n && t.b<n*n && t.c<n*n)
            sparsegamma.push_back((TensorEntry){(short)t.a, (short)t.b, (short)t.c, t.k});
    }
    sparsefile.close();
    return sparsegamma;
}


std::vector<TensorEntry> gpu_filterGamma(std::vector<TensorEntry> v, int a, int b, int c)
{
    std::vector<TensorEntry> res;
    for (auto e: v) {
        if (e.a < a*a && e.b < b*b && e.c < c*c)
            res.push_back(e);
    }
    return res;
}

void gpu_initGamma()
{
    std::vector<TensorEntry> v,v1;
    int size = 0;
    TensorEntry* p;
    // load sparse gamma n
    v = readGamma(n);
    size = v.size();
    //console.log("sparse n size:", size);
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGammaSize, &size, sizeof(int));
    gpuErrchk( cudaPeekAtLastError() );
    // load sparse gamma 2n-1
    v = readGamma(3*n-2);
    size = v.size();
    //console.log("sparse 3n-2 size:", size);
    // gamma 1,1,2
    v1 = gpu_filterGamma(v, n, n, 2*n-1);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma112, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma112Size, &size, sizeof(int));
    // gamma 2,1,3
    v1 = gpu_filterGamma(v, 2*n-1, n, 3*n-2);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma213, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma213Size, &size, sizeof(int));
    // set gamma on CPU
    gpuErrchk( cudaPeekAtLastError() );
    SparseGamma3 = v;
    //
    v = readGamma(4*n-3);
    // gamma 3,1,4
    v1 = gpu_filterGamma(v, 3*n-2, n, 4*n-3);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma314, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma314Size, &size, sizeof(int));
    // gamma 4,1,1
    v1 = gpu_filterGamma(v, 4*n-3, n, n);
    size = v1.size();
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v1[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma411, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGamma411Size, &size, sizeof(int));
}

void releaseGamma()
{
    TensorEntry* p;
    cudaMemcpyFromSymbol(&p, deviceSparseGamma, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma112, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma213, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma314, sizeof(TensorEntry*));
    cudaFree(p);
    cudaMemcpyFromSymbol(&p, deviceSparseGamma411, sizeof(TensorEntry*));
    cudaFree(p);
}

// convert from n*n SH vector to coefficients of Fourier Series
// placed at lower-most corner in the N*N array
__global__ void cu_sh1_fs5(float* SH, cufftComplex* FS)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N5*N5;
    // copy to register
    float SHreg[n*n];
    memcpy(SHreg, SH+shbase, n*n*sizeof(float));
    // execute
    #include "generated/sh1_fs5.cu"
}

__global__ void cu_fs2sh(cufftComplex* FS, float* SH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N2*N2;
    // copy to register
    float SHreg[n*n];
    // execute
    #include "generated/fs2sh.cu"
    // copy back to global memory
    memcpy(SH+shbase, SHreg, n*n*sizeof(float));
}

// convert from coefficients of Fourier Series to SH vector
__global__ void cu_fs5_sh1(cufftComplex* FS, float* SH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N5*N5;
    // copy to register
    float SHreg[n*n];
    #include "generated/fs5_sh1.cu"
    // copy back to global memory
    memcpy(SH+shbase, SHreg, n*n*sizeof(float));
}

__global__ void cu_sh2fs(float* SH, cufftComplex* FS)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N2*N2;
    // copy to register
    float SHreg[n*n];
    memcpy(SHreg, SH+shbase, n*n*sizeof(float));
    #include "generated/sh2fs.cu"
}

// element-wise multiplication B_i *= A_i
__global__ void multiply(cufftComplex* A, cufftComplex* B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = A[i].x * B[i].x - A[i].y * B[i].y;
	float y = A[i].y * B[i].x + A[i].x * B[i].y;
	B[i].x = x;
	B[i].y = y;
}

// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...
void shprod_many(float* A, float* B, float* C, float* D, float* E, float* F,
            cufftComplex* pool0, cufftComplex* pool1, cufftComplex* pool2,
            int multi_product_num, cufftHandle plan)
{
	const int blocksize = 32;
	assert(multi_product_num%blocksize == 0);
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(A, pool0);

	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);

	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(B, pool0);

	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	multiply<<<multi_product_num*N5*N5/blocksize, blocksize>>>(pool1, pool2);

	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(C, pool0);

	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N5*N5/blocksize, blocksize>>>(pool1, pool2);

	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(D, pool0);

	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N5*N5/blocksize, blocksize>>>(pool1, pool2);

	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(E, pool0);

	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N5*N5/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);

	cu_fs5_sh1<<<multi_product_num/blocksize, blocksize>>>(pool1, F);
	// synchronize
	cudaDeviceSynchronize();
	//console.log("sh2fsexec:", dtsh);
	//console.log("fftexec:", fftdt);
	//console.log("fs2shexec:", dt-fftdt);
    //console.timeEnd("exclude_planning " + std::to_string(num));
}

__global__ void shprod_conventional(float* A, float* B, float* C, float* D, float* E, float* F)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n*n];
    float Breg[n*n];
    float Treg[n*n];
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    memset(Treg, 0, n*n*sizeof(float));
#define e deviceSparseGamma[i]
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Treg[e.c] += e.val * Areg[e.a] * Breg[e.b];
    memcpy(Areg, C+base, n*n*sizeof(float));
    memset(Breg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Breg[e.c] += e.val * Treg[e.a] * Areg[e.b];
    memcpy(Areg, D+base, n*n*sizeof(float));
    memset(Treg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Treg[e.c] += e.val * Areg[e.a] * Breg[e.b];
    memcpy(Areg, E+base, n*n*sizeof(float));
    memset(Breg, 0, n*n*sizeof(float));
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Breg[e.c] += e.val * Treg[e.a] * Areg[e.b];
#undef e
    memcpy(F+base, Breg, n*n*sizeof(float));
}


__global__ void shprod_conventional_precise(float* A, float* B, float* C, float* D, float* E, float* F)
{
    // bruteforce, not yet optimized
    constexpr int n1 = 4*n-3;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n1*n1];
    float Breg[n1*n1];
    float T1reg[n1*n1];
    float T2reg[n1*n1];
    memset(Areg, 0, n1*n1*sizeof(float));
    memset(Breg, 0, n1*n1*sizeof(float));
#define SHMUL(A,B,C,ka,kb,kc) \
    memset(C, 0, n1*n1*sizeof(float)); \
    for (int i=0; i<deviceSparseGamma##ka##kb##kc##Size; ++i) \
        C[deviceSparseGamma##ka##kb##kc[i].c] += deviceSparseGamma##ka##kb##kc[i].val * A[deviceSparseGamma##ka##kb##kc[i].a] * B[deviceSparseGamma##ka##kb##kc[i].b];
    // T2 = A * B
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    SHMUL(Areg, Breg, T2reg, 1,1,2);
    // T1 = T2 * C
    memcpy(Areg, C+base, n*n*sizeof(float));
    SHMUL(T2reg, Areg, T1reg, 2,1,3);
    // T2 = T1 * D
    memcpy(Areg, D+base, n*n*sizeof(float));
    SHMUL(T1reg, Areg, T2reg, 3,1,4);
    // Areg = T2 * E
    memcpy(Breg, E+base, n*n*sizeof(float));
    SHMUL(T2reg, Breg, Areg, 4,1,1);
    memcpy(F+base, Areg, n*n*sizeof(float));
}

void multi_product(float *A, float *B, float* C, float *D, float *E, float *F,
                    int multi_product_num, int type)
{
    assert(multi_product_num%traditional_blocksize==0);
    dim3 grid(multi_product_num/traditional_blocksize,1);
    dim3 block(traditional_blocksize,1);

    if(type == 0)shprod_conventional<<<grid, block>>>(A,B,C,D,E,F);
    else if(type == 1)shprod_conventional_precise<<<grid, block>>>(A,B,C,D,E,F);
}

void shprod_many(float* A, float* B, float* C, 
                cufftComplex *pool0, cufftComplex *pool1, cufftComplex *pool2,
                int num, cufftHandle plan)
{
	const int blocksize = 32;
	assert(num%blocksize == 0);

    cudaMemset(pool0, 0, sizeof(cufftComplex)*N2*N2*num);

	cu_sh2fs<<<num/blocksize, blocksize>>>(A, pool0);
    cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);

    cu_sh2fs<<<num/blocksize, blocksize>>>(B, pool0);
	//cudaDeviceSynchronize();

	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	// element-wise multiply
	multiply<<<num*N2*N2/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);
	//cudaDeviceSynchronize();

	cu_fs2sh<<<num/blocksize, blocksize>>>(pool1, C);
	//cudaDeviceSynchronize();
}
