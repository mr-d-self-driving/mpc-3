/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) car_kinematic_cost_ext_cost_0_fun_jac_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sign CASADI_PREFIX(sign)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

static const casadi_int casadi_s0[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s1[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s4[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};

/* car_kinematic_cost_ext_cost_0_fun_jac:(i0[6],i1[3],i2[])->(o0,o1[9]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a4, a5, a6, a7, a8, a9;
  a0=5.;
  a1=-1.3845493629437283e-01;
  a2=arg[0]? arg[0][5] : 0;
  a3=(a1*a2);
  a4=5.9925004824816143e-02;
  a3=(a3+a4);
  a4=(a3*a2);
  a5=2.0059897207952213e-01;
  a4=(a4+a5);
  a5=(a4*a2);
  a6=2.3086484313215150e-01;
  a5=(a5+a6);
  a6=(a5*a2);
  a7=1.5878559642148511e+00;
  a6=(a6+a7);
  a8=(a6*a2);
  a9=1.3592101520430320e+00;
  a8=(a8+a9);
  a9=arg[0]? arg[0][0] : 0;
  a8=(a8-a9);
  a9=-6.9227468147186411e-01;
  a10=(a9*a2);
  a11=2.3970001929926457e-01;
  a10=(a10+a11);
  a11=(a10*a2);
  a12=6.0179691623856635e-01;
  a11=(a11+a12);
  a12=(a11*a2);
  a13=4.6172968626430300e-01;
  a12=(a12+a13);
  a13=(a12*a2);
  a13=(a13+a7);
  a7=casadi_sq(a13);
  a14=-3.9562898318218451e+00;
  a15=(a14*a2);
  a16=-3.7212344094208509e+00;
  a15=(a15+a16);
  a16=(a15*a2);
  a17=-8.3337619914141525e-01;
  a16=(a16+a17);
  a17=(a16*a2);
  a18=-7.7056638584115789e-01;
  a17=(a17+a18);
  a18=(a17*a2);
  a19=2.0690500327448431e+00;
  a18=(a18+a19);
  a20=casadi_sq(a18);
  a7=(a7+a20);
  a7=sqrt(a7);
  a20=(a13/a7);
  a21=(a8*a20);
  a22=-7.9125796636436907e-01;
  a23=(a22*a2);
  a24=-9.3030860235521273e-01;
  a23=(a23+a24);
  a24=(a23*a2);
  a25=-2.7779206638047177e-01;
  a24=(a24+a25);
  a25=(a24*a2);
  a26=-3.8528319292057894e-01;
  a25=(a25+a26);
  a26=(a25*a2);
  a26=(a26+a19);
  a19=(a26*a2);
  a27=2.3155917952757923e+00;
  a19=(a19+a27);
  a27=arg[0]? arg[0][1] : 0;
  a19=(a19-a27);
  a27=(a18/a7);
  a28=(a19*a27);
  a21=(a21+a28);
  a28=(a21*a20);
  a28=(a8-a28);
  a29=casadi_sq(a28);
  a30=(a21*a27);
  a30=(a19-a30);
  a31=casadi_sq(a30);
  a29=(a29+a31);
  a29=(a0*a29);
  a31=fabs(a21);
  a32=casadi_sq(a31);
  a29=(a29+a32);
  a32=5.0000000000000000e-01;
  a33=1.;
  a33=(a2-a33);
  a34=casadi_sq(a33);
  a34=(a32*a34);
  a29=(a29+a34);
  a34=arg[1]? arg[1][0] : 0;
  a35=casadi_sq(a34);
  a29=(a29+a35);
  a35=arg[1]? arg[1][1] : 0;
  a36=casadi_sq(a35);
  a29=(a29+a36);
  a36=arg[1]? arg[1][2] : 0;
  a37=casadi_sq(a36);
  a29=(a29+a37);
  if (res[0]!=0) res[0][0]=a29;
  a34=(a34+a34);
  if (res[1]!=0) res[1][0]=a34;
  a35=(a35+a35);
  if (res[1]!=0) res[1][1]=a35;
  a36=(a36+a36);
  if (res[1]!=0) res[1][2]=a36;
  a28=(a28+a28);
  a28=(a0*a28);
  a36=casadi_sign(a21);
  a31=(a31+a31);
  a36=(a36*a31);
  a30=(a30+a30);
  a0=(a0*a30);
  a30=(a27*a0);
  a36=(a36-a30);
  a30=(a20*a28);
  a36=(a36-a30);
  a30=(a20*a36);
  a30=(a28+a30);
  a31=(-a30);
  if (res[1]!=0) res[1][3]=a31;
  a31=(a27*a36);
  a31=(a0+a31);
  a35=(-a31);
  if (res[1]!=0) res[1][4]=a35;
  a35=0.;
  if (res[1]!=0) res[1][5]=a35;
  if (res[1]!=0) res[1][6]=a35;
  if (res[1]!=0) res[1][7]=a35;
  a33=(a33+a33);
  a32=(a32*a33);
  a26=(a26*a31);
  a32=(a32+a26);
  a31=(a2*a31);
  a25=(a25*a31);
  a32=(a32+a25);
  a31=(a2*a31);
  a24=(a24*a31);
  a32=(a32+a24);
  a31=(a2*a31);
  a23=(a23*a31);
  a32=(a32+a23);
  a31=(a2*a31);
  a22=(a22*a31);
  a32=(a32+a22);
  a19=(a19*a36);
  a0=(a21*a0);
  a19=(a19-a0);
  a0=(a19/a7);
  a18=(a18+a18);
  a27=(a27/a7);
  a27=(a27*a19);
  a20=(a20/a7);
  a8=(a8*a36);
  a21=(a21*a28);
  a8=(a8-a21);
  a20=(a20*a8);
  a27=(a27+a20);
  a20=(a7+a7);
  a27=(a27/a20);
  a18=(a18*a27);
  a0=(a0-a18);
  a17=(a17*a0);
  a32=(a32+a17);
  a0=(a2*a0);
  a16=(a16*a0);
  a32=(a32+a16);
  a0=(a2*a0);
  a15=(a15*a0);
  a32=(a32+a15);
  a0=(a2*a0);
  a14=(a14*a0);
  a32=(a32+a14);
  a8=(a8/a7);
  a13=(a13+a13);
  a13=(a13*a27);
  a8=(a8-a13);
  a12=(a12*a8);
  a32=(a32+a12);
  a8=(a2*a8);
  a11=(a11*a8);
  a32=(a32+a11);
  a8=(a2*a8);
  a10=(a10*a8);
  a32=(a32+a10);
  a8=(a2*a8);
  a9=(a9*a8);
  a32=(a32+a9);
  a6=(a6*a30);
  a32=(a32+a6);
  a30=(a2*a30);
  a5=(a5*a30);
  a32=(a32+a5);
  a30=(a2*a30);
  a4=(a4*a30);
  a32=(a32+a4);
  a30=(a2*a30);
  a3=(a3*a30);
  a32=(a32+a3);
  a2=(a2*a30);
  a1=(a1*a2);
  a32=(a32+a1);
  if (res[1]!=0) res[1][8]=a32;
  return 0;
}

CASADI_SYMBOL_EXPORT int car_kinematic_cost_ext_cost_0_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int car_kinematic_cost_ext_cost_0_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int car_kinematic_cost_ext_cost_0_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void car_kinematic_cost_ext_cost_0_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int car_kinematic_cost_ext_cost_0_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void car_kinematic_cost_ext_cost_0_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void car_kinematic_cost_ext_cost_0_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void car_kinematic_cost_ext_cost_0_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int car_kinematic_cost_ext_cost_0_fun_jac_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int car_kinematic_cost_ext_cost_0_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real car_kinematic_cost_ext_cost_0_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* car_kinematic_cost_ext_cost_0_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* car_kinematic_cost_ext_cost_0_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* car_kinematic_cost_ext_cost_0_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* car_kinematic_cost_ext_cost_0_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int car_kinematic_cost_ext_cost_0_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 3;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif