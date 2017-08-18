#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include <cblas.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"
#include "stdio.h"
#ifdef __cplusplus
extern "C" {
#endif

using std::string;
using std::vector;
using caffe::CaffeMobile;

float addFloat(float a,float b)
  {
    return a+b;
  }

int getTimeSec() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int)now.tv_sec;
}

//===================================================UNITY3D
cv::Mat imgbuf2matOfUnity(uint8_t *buf, int width, int height) {
if (!buf)
  {
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is NULL");
  }else{
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is not NULL");
  }
  // cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)buf);
 // cv::cvtColor(img, img, CV_YUV2BGR_NV21);
  // cv::cvtColor(img, img, CV_BGRA2BGR,3);
  cv::Mat img;
  cv::Mat img_ = cv::Mat(480, 640, CV_8UC(4), (uchar*)buf);

    cv::cvtColor(img_, img, CV_RGBA2BGR);
    cv::flip(img,img,0);
  return img;
}
cv::Mat getImageOfUnity(uint8_t *buf, int width, int height) {
  return (width == 0 && height == 0) ? cv::imread((char*) buf, -1)
                                     : imgbuf2matOfUnity(buf, width, height);
}
void setNumThreadsForUnity(int num_threads){
  openblas_set_num_threads(num_threads);
}

void SetMeanWithMeanFileForUnity(const char* meanFile,int type){
  __android_log_print(ANDROID_LOG_INFO,"JNI",meanFile);
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  caffe_mobile->SetMean(meanFile);
}


int loadModelForUnity(const char* modelPath,const char* weightsPath, int type){
  __android_log_print(ANDROID_LOG_INFO,"JNI",modelPath);
  CaffeMobile::Get(modelPath,weightsPath,type);
  return 0;
}

void setMeanWithMeanValuesForUnity(float* meanValues,int type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<float> vMeanValues(meanValues,meanValues+sizeof(meanValues)/sizeof(float));
  caffe_mobile->SetMean(vMeanValues);
}


void setScaleForUnity(float scale,int type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  caffe_mobile->SetScale(scale);
}

void getConfidenceScoreForUnity(uint8_t *buf,int width,int height,int type,float *pScore) {

if (!buf)
  {
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is NULL");
  }else{
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is not NULL");
  }

  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<float> conf_score =caffe_mobile->GetConfidenceScore(getImageOfUnity(buf, width, height));

  if(!pScore)
    return;
  pScore[0] = conf_score[0];
  pScore[1] = conf_score[1];
  pScore[2] = conf_score[2];
  pScore[3] = conf_score[3];
  //sprintf(cStr,"conf_score=%f,%f,%f,%f",conf_score[0],conf_score[1],conf_score[2],conf_score[3]);
  //__android_log_print(ANDROID_LOG_INFO,"JNI",cStr);

}

void predictImageForUnity(uint8_t *buf, int width, int height,int k,int type, int* pTopK) {

  if (!buf)
  {
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is NULL");
  }else{
    __android_log_print(ANDROID_LOG_INFO,"JNI","buff is not NULL");
  }

  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<int> top_k =
      caffe_mobile->PredictTopK(getImageOfUnity(buf, width, height), k);
  char cStr[256];
  sprintf(cStr,"top_k[0]=%d",top_k[0]);
  __android_log_print(ANDROID_LOG_INFO,"JNI",cStr);

  if(!pTopK)
    return;
  pTopK[0] = top_k[0];
  sprintf(cStr,"pTopK[0]=%d",pTopK[0]);
  __android_log_print(ANDROID_LOG_INFO,"JNI",cStr);
}
//===================================================ANDROID
string jstring2string(JNIEnv *env, jstring jstr) {
  const char *cstr = env->GetStringUTFChars(jstr, 0);
  string str(cstr);
  env->ReleaseStringUTFChars(jstr, cstr);
  return str;
}

/**
 * NOTE: byte[] buf = str.getBytes("US-ASCII")
 */
string bytes2string(JNIEnv *env, jbyteArray buf) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  string s((char *)ptr, env->GetArrayLength(buf));
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return s;
}

cv::Mat imgbuf2mat(JNIEnv *env, jbyteArray buf, int width, int height) {
  jbyte *ptr = env->GetByteArrayElements(buf, 0);
  cv::Mat img(height + height / 2, width, CV_8UC1, (unsigned char *)ptr);
  cv::cvtColor(img, img, CV_YUV2BGR_NV21);
  env->ReleaseByteArrayElements(buf, ptr, 0);
  return img;
}


cv::Mat getImage(JNIEnv *env, jbyteArray buf, int width, int height) {
  return (width == 0 && height == 0) ? cv::imread(bytes2string(env, buf), -1)
                                     : imgbuf2mat(env, buf, width, height);
}


JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setNumThreads(JNIEnv *env,
                                                             jobject thiz,
                                                             jint numThreads) {
  int num_threads = numThreads;
  openblas_set_num_threads(num_threads);
}


JNIEXPORT void JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_enableLog(
    JNIEnv *env, jobject thiz, jboolean enabled) {}

JNIEXPORT jint JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_loadModel(
    JNIEnv *env, jobject thiz, jstring modelPath, jstring weightsPath,jint type) {
  CaffeMobile::Get(jstring2string(env, modelPath),
                   jstring2string(env, weightsPath),type);
  return 0;
}


JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanFile(
    JNIEnv *env, jobject thiz, jstring meanFile,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  caffe_mobile->SetMean(jstring2string(env, meanFile));
}

JNIEXPORT void JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setMeanWithMeanValues(
    JNIEnv *env, jobject thiz, jfloatArray meanValues,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  int num_channels = env->GetArrayLength(meanValues);
  jfloat *ptr = env->GetFloatArrayElements(meanValues, 0);
  vector<float> mean_values(ptr, ptr + num_channels);
  caffe_mobile->SetMean(mean_values);
}


JNIEXPORT void JNICALL Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_setScale(
    JNIEnv *env, jobject thiz, jfloat scale,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  caffe_mobile->SetScale(scale);
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jfloatArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_getConfidenceScore(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<float> conf_score =
      caffe_mobile->GetConfidenceScore(getImage(env, buf, width, height));

  jfloatArray result;
  result = env->NewFloatArray(conf_score.size());
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }
  // move from the temp structure to the java structure
  env->SetFloatArrayRegion(result, 0, conf_score.size(), &conf_score[0]);
  return result;
}

/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jintArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_predictImage(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height,
    jint k,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<int> top_k =
      caffe_mobile->PredictTopK(getImage(env, buf, width, height), k);

  jintArray result;
  result = env->NewIntArray(k);
  if (result == NULL) {
    return NULL; /* out of memory error thrown */
  }
  // move from the temp structure to the java structure
  env->SetIntArrayRegion(result, 0, k, &top_k[0]);
  return result;
}



/**
 * NOTE: when width == 0 && height == 0, buf is a byte array
 * (str.getBytes("US-ASCII")) which contains the img path
 */
JNIEXPORT jobjectArray JNICALL
Java_com_sh1r0_caffe_1android_1lib_CaffeMobile_extractFeatures(
    JNIEnv *env, jobject thiz, jbyteArray buf, jint width, jint height,
    jstring blobNames,jint type) {
  CaffeMobile *caffe_mobile = CaffeMobile::Get(type);
  vector<vector<float>> features = caffe_mobile->ExtractFeatures(
      getImage(env, buf, width, height), jstring2string(env, blobNames));

  jobjectArray array2D =
      env->NewObjectArray(features.size(), env->FindClass("[F"), NULL);
  for (size_t i = 0; i < features.size(); ++i) {
    jfloatArray array1D = env->NewFloatArray(features[i].size());
    if (array1D == NULL) {
      return NULL; /* out of memory error thrown */
    }
    // move from the temp structure to the java structure
    env->SetFloatArrayRegion(array1D, 0, features[i].size(), &features[i][0]);
    env->SetObjectArrayElement(array2D, i, array1D);
  }
  return array2D;
}


JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env = NULL;
  jint result = -1;

  if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
    LOG(FATAL) << "GetEnv failed!";
    return result;
  }

  FLAGS_redirecttologcat = true;
  FLAGS_android_logcat_tag = "caffe_jni";

  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
