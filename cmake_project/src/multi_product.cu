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
/*
static std::vector<TensorEntry> readGamma_ascii(int n)
{
    std::vector<TensorEntry> sparsegamma;
    std::string line;
    std::ifstream sparsefile("../gamma/sparse" + std::to_string(n));
    TensorEntry entry;
    while(getline(sparsefile, line))
    {
        std::vector<std::string> tokens = split(line.substr(1, line.length() - 3), ',');
        entry.a = std::stoi(tokens[0]);
        entry.b = std::stoi(tokens[1]);
        entry.c = std::stoi(tokens[2]);
        entry.val = std::stof(tokens[3]);
        sparsegamma.push_back(entry);
    }
    sparsefile.close();
    return sparsegamma;
}*/

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
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
//    cufftComplex FSreg[N*N];
    memcpy(SHreg, SH+shbase, n*n*sizeof(float));
//    memset(FSreg, 0, N*N*sizeof(cufftComplex));
    // execute
    #include "generated/sh1_fs5.cu"
    // copy back to global memory
//   	for (int j=0; j<N*N; ++j)
//   		FS[j+i*N*N] = FSreg[j];
}

// convert from coefficients of Fourier Series to SH vector
__global__ void cu_fs5_sh1(cufftComplex* FS, float* SH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
//    cufftComplex FSreg[N*N];
//    memset(SHreg, 0, n*n*sizeof(float));
//   	for (int j=0; j<N*N; ++j)
//   		FSreg[j] = FS[j+i*N*N];
    // execute
    #include "generated/fs5_sh1.cu"
    // copy back to global memory
    memcpy(SH+shbase, SHreg, n*n*sizeof(float));
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
//	auto t0 = std::chrono::system_clock::now();
//	auto t1 = std::chrono::system_clock::now();
//	double dt = 0;
//	double dtsh = 0;
//#define STARTTIME {cudaDeviceSynchronize(); t0 = std::chrono::system_clock::now();}
//#define ENDTIME {cudaDeviceSynchronize(); t1 = std::chrono::system_clock::now(); dt += std::chrono::duration<double>(t1-t0).count()*1000;}
//#define ENDTIMESH {cudaDeviceSynchronize(); t1 = std::chrono::system_clock::now(); dtsh += std::chrono::duration<double>(t1-t0).count()*1000;}
	const int blocksize = 32;
	assert(multi_product_num%blocksize == 0);
	// mem alloc
	//cufftComplex *pool0, *pool1, *pool2;
	//cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*num);
	//cudaMalloc((void**)&pool1, sizeof(cufftComplex)*N*N*num);
	//cudaMalloc((void**)&pool2, sizeof(cufftComplex)*N*N*num);
	// plan DFT
	//cufftHandle plan;
	//int sizes[2] = {N,N};
	//cufftPlanMany(&plan, 2, sizes, NULL, 1, N*N, NULL, 1, N*N, CUFFT_C2C, multi_product_num);
    //console.time("exclude_planning " + std::to_string(num));

//STARTTIME
	// DFT on A
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(A, pool0);
//ENDTIMESH

//STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
//ENDTIME

//STARTTIME
	// DFT on B & multiply
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(B, pool0);
//ENDTIMESH

//STARTTIME
	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	multiply<<<multi_product_num*N*N/blocksize, blocksize>>>(pool1, pool2);
//ENDTIME

//STARTTIME
	// DFT on C & multiply
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(C, pool0);
//ENDTIMESH

//STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N*N/blocksize, blocksize>>>(pool1, pool2);
//ENDTIME

//STARTTIME
	// DFT on D & multiply
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(D, pool0);
//ENDTIMESH

//STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N*N/blocksize, blocksize>>>(pool1, pool2);
//ENDTIME

//STARTTIME
	// DFT on E & multiply
	cu_sh1_fs5<<<multi_product_num/blocksize, blocksize>>>(E, pool0);
//ENDTIMESH

//STARTTIME
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	multiply<<<multi_product_num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);
//ENDTIME
//	double fftdt = dt;
//STARTTIME
	cu_fs5_sh1<<<multi_product_num/blocksize, blocksize>>>(pool1, F);
//ENDTIME
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


/*void validate_sh1_fs5(float* deviceA)
{
    cufftComplex *pool0;
    cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*(num+1));
    float* B;
    cudaMalloc((void**)&B, sizeof(float)*n*n*num);
    // call
    const int blocksize = 32;
    const int extrasize = ((4*n-4)*N+(4*n-4)) - ((n-1)*N+(n-1));
    cu_sh1_fs5<<<num/blocksize, blocksize>>>(deviceA, pool0+extrasize);
    cudaMemset(pool0, 0, extrasize * sizeof(cufftComplex));
    gpuErrchk(cudaDeviceSynchronize());
    cu_fs5_sh1<<<num/blocksize, blocksize>>>(pool0, B);
    gpuErrchk(cudaDeviceSynchronize());
    // validate
    float* p = new float[n*n];
    cudaMemcpy(p, deviceA, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i] << " ";
    puts("");
    cudaMemcpy(p, B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i=0; i<n*n; ++i)
        std::cout << p[i]*N*N << " ";
    puts("");
}*/


/*void compute_reference(float* A, float* B, float* C, float* D, float* E, float* res)
{
    // SH<3*n-2> A1,B1,C1,D1,E1;
    // memcpy(A1.a, A, n*n*sizeof(float));
    // memcpy(B1.a, B, n*n*sizeof(float));
    // memcpy(C1.a, C, n*n*sizeof(float));
    // memcpy(D1.a, D, n*n*sizeof(float));
    // memcpy(E1.a, E, n*n*sizeof(float));
    // SH<3*n-2> ref = A1*B1*C1*D1*E1;
    // memcpy(res, ref.a, n*n*sizeof(float));
    // return;

    // compute reference on CPU, un-optimized
    // convert them all to (3n-2) order
    const int n1 = 3*n-2; /////// TODO
    float A1[n1*n1], B1[n1*n1], C1[n1*n1], D1[n1*n1], E1[n1*n1];
    float M1[n1*n1], M2[n1*n1], M3[n1*n1];
    memset(A1, 0, n1*n1*sizeof(float));
    memset(B1, 0, n1*n1*sizeof(float));
    memset(C1, 0, n1*n1*sizeof(float));
    memset(D1, 0, n1*n1*sizeof(float));
    memset(E1, 0, n1*n1*sizeof(float));
    memset(M1, 0, n1*n1*sizeof(float));
    memset(M2, 0, n1*n1*sizeof(float));
    memset(M3, 0, n1*n1*sizeof(float));
    memcpy(A1, A, n*n*sizeof(float));
    memcpy(B1, B, n*n*sizeof(float));
    memcpy(C1, C, n*n*sizeof(float));
    memcpy(D1, D, n*n*sizeof(float));
    memcpy(E1, E, n*n*sizeof(float));
    // M2 = A1 * B1
    for (auto e: SparseGamma3)
        M2[e.c] += e.val * A1[e.a] * B1[e.b];
    // M1 = M1 * C1
    for (auto e: SparseGamma3)
        M1[e.c] += e.val * M2[e.a] * C1[e.b];
    // M2 = D1 * E1
    memset(M2, 0, n1*n1*sizeof(float));
    for (auto e: SparseGamma3)
        M2[e.c] += e.val * D1[e.a] * E1[e.b];
    // M3 = M1 * M2 (order matters!)
    for (auto e: SparseGamma3)
        M3[e.c] += e.val * M1[e.a] * M2[e.b];
    // copy result
    memcpy(res, M3, n*n*sizeof(float));
}*/

/* return relative error of kth result
float validate_err(float* deviceA, float* deviceB, float* deviceC, float* deviceD, float* deviceE, float* deviceF, int k)
{blocksize
    float sum = 0;
    const int n_sample = 10;
    for (int i=0; i<n_sample; ++i)
        sum += validate_err(deviceA, deviceB, deviceC, deviceD, deviceE, deviceF, rand()%num);
    float err_avg = sum / n_sample;
    console.log("err:", err_avg);
}*/



void multi_product(float *A, float *B, float* C, float *D, float *E, float *F,
                    int multi_product_num, int type)
{
    assert(multi_product_num%traditional_blocksize==0);
    dim3 grid(multi_product_num/traditional_blocksize,1);
    dim3 block(traditional_blocksize,1);

    if(type == 0)shprod_conventional<<<grid, block>>>(A,B,C,D,E,F);
    else if(type == 1)shprod_conventional_precise<<<grid, block>>>(A,B,C,D,E,F);
}
