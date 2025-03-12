#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <ATen/Dispatch.h>

template <typename T> struct OpType {
    typedef T type;
};

template <> struct OpType<__nv_bfloat16> {
    typedef float type;
};

template <> struct OpType<__half> {
    typedef float type;
};

template <> struct OpType<c10::Half> {
    typedef float type;
};

template <> struct OpType<c10::BFloat16> {
    typedef float type;
};